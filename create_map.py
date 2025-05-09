import os
import numpy as np
import cv2
import tqdm
from base64 import b64encode
import torch
import math
from tqdm import tqdm
from torchvision.transforms import transforms
import matplotlib.pyplot as plt
from PIL import Image

from utils import *
import clip
from lseg.modules.models.lseg_net import LSegEncNet


clip_version = "ViT-B/32"
clip_feat_dim = {'RN50': 1024, 'RN101': 512, 'RN50x4': 640, 'RN50x16': 768,
                'RN50x64': 1024, 'ViT-B/32': 512, 'ViT-B/16': 512, 'ViT-L/14': 768}[clip_version]
checkpoint_dir='demo_e200.ckpt'

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")


def load_scene_path(data_dir):
    rgb_dir = os.path.join(data_dir, "color")
    depth_dir = os.path.join(data_dir, "depth")
    pose_dir = os.path.join(data_dir, "pose")

    # check if folders exist
    if not (os.path.exists(rgb_dir) and os.path.exists(depth_dir) and os.path.exists(pose_dir)):
        raise FileNotFoundError(f"color, depth, or pose folders are missing in {data_dir}")

    rgb_list = sorted([os.path.join(rgb_dir, f) for f in os.listdir(rgb_dir)])
    depth_list = sorted([os.path.join(depth_dir, f) for f in os.listdir(depth_dir)])
    pose_list = sorted([os.path.join(pose_dir, f) for f in os.listdir(pose_dir)])

    return rgb_list, depth_list, pose_list

def load_lseg_model(checkpoint_dir, labels, crop_size):
    model = LSegEncNet(labels, arch_option=0, block_depth=0, activation='lrelu', crop_size=crop_size)

    # Load checkpoint (allowing full object because old model)
    pretrained_state_dict = torch.load(checkpoint_dir, weights_only=False, map_location=device)

    # Correct: use pretrained_state_dict, not checkpoint
    pretrained_state_dict = {k.lstrip('net.'): v for k, v in pretrained_state_dict['state_dict'].items()}

    model.load_state_dict(pretrained_state_dict)
    model.eval()
    return model

def get_lseg_feat(
    model,
    image,
    labels,
    transform,
    crop_size=480,
    base_size=520,
    norm_mean=[0.5, 0.5, 0.5],
    norm_std=[0.5, 0.5, 0.5]
):
    '''
    Args:
    - model: LSeg model
    - image: RGB image (H, W, C)
    - labels: list of text labels
    - transform: torchvision transforms.Compose
    - crop_size: crop size for sliding window
    - base_size: long side resize
    - norm_mean, norm_std: normalization for padding

    Returns:
    - outputs: numpy array (B, D, H, W)
    '''
    device = next(model.parameters()).device
    image = transform(image).unsqueeze(0).to(device)  # (1, C, H, W)

    batch_size, _, original_height, original_width = image.size()
    stride = int(crop_size * (2.0/3.0))  # sliding stride

    # Resize maintaining aspect ratio
    if original_height > original_width:
        resized_height = base_size
        resized_width = int(base_size * original_width / original_height + 0.5)
    else:
        resized_width = base_size
        resized_height = int(base_size * original_height / original_width + 0.5)

    resized_image = resize_image(
        image, resized_height, resized_width,
        mode='bilinear', align_corners=True
    )

    # Case 1: resized image fits in one crop
    if max(resized_height, resized_width) <= crop_size:
        padded_image = pad_image(resized_image, norm_mean, norm_std, crop_size)
        with torch.no_grad():
            features, logits = model(padded_image, labels)
        features = crop_image(features, 0, resized_height, 0, resized_width)

    # Case 2: need sliding window
    else:
        if min(resized_height, resized_width) < crop_size:
            padded_image = pad_image(resized_image, norm_mean, norm_std, crop_size)
        else:
            padded_image = resized_image

        _, _, padded_height, padded_width = padded_image.shape

        num_vertical_steps = math.ceil((padded_height - crop_size) / stride) + 1
        num_horizontal_steps = math.ceil((padded_width - crop_size) / stride) + 1

        with torch.cuda.device_of(image):
            with torch.no_grad():
                features = torch.zeros(batch_size, model.out_c, padded_height, padded_width, device=device)
                logits_accumulated = torch.zeros(batch_size, len(labels), padded_height, padded_width, device=device)
                count_normalizer = torch.zeros(batch_size, 1, padded_height, padded_width, device=device)

        # Sliding window evaluation
        for vertical_idx in range(num_vertical_steps):
            for horizontal_idx in range(num_horizontal_steps):
                top = vertical_idx * stride
                left = horizontal_idx * stride
                bottom = min(top + crop_size, padded_height)
                right = min(left + crop_size, padded_width)

                cropped_input = crop_image(padded_image, top, bottom, left, right)
                cropped_input_padded = pad_image(cropped_input, norm_mean, norm_std, crop_size)

                with torch.no_grad():
                    cropped_features, cropped_logits = model(cropped_input_padded, labels)

                cropped_features = crop_image(cropped_features, 0, bottom-top, 0, right-left)
                cropped_logits = crop_image(cropped_logits, 0, bottom-top, 0, right-left)

                features[:, :, top:bottom, left:right] += cropped_features
                logits_accumulated[:, :, top:bottom, left:right] += cropped_logits
                count_normalizer[:, :, top:bottom, left:right] += 1

        assert (count_normalizer == 0).sum() == 0, "Some pixels were never visited."

        features = features / count_normalizer
        logits_accumulated = logits_accumulated / count_normalizer

        features = features[:, :, :resized_height, :resized_width]
        logits_accumulated = logits_accumulated[:, :, :resized_height, :resized_width]

    features = features.cpu().numpy()  # (B, D, H, W)

    # Optionally extract predicted segmentation (not returned)
    predicts = [torch.max(logit, 0)[1].cpu().numpy() for logit in logits_accumulated]
    pred_mask = predicts[0]

    return features

def get_text_feat(clip_model, in_text, clip_dim, batch_size=64 ):
    '''
    Args:
    CLIP text encoder
    in_text: list of texts
    clip_dim: dimension of CLIP text feature
    batch_size: batch size for encoding
    Returns:
    text_feat: numpy array of text features. shape: (len(in_text), clip_dim)
    '''
    if torch.cuda.is_available():
        text_tokens = clip.tokenize(in_text).cuda()
    elif torch.backends.mps.is_available():
        text_tokens = clip.tokenize(in_text).to("mps")
    else:
        text_tokens = clip.tokenize(in_text)

    text_id = 0
    text_feat = np.zeros((len(in_text), clip_dim),dtype=np.float32)

    while text_id < len(in_text):
        batch_size = min(len(in_text) - text_id, batch_size)
        text_tokens_batch = text_tokens[text_id:text_id+batch_size]
        with torch.no_grad():
            batch_feats = clip_model.encode_text(text_tokens_batch).float()
        batch_feats /= batch_feats.norm(dim=-1, keepdim=True)
        text_feat[text_id:text_id+batch_size] = batch_feats.cpu().numpy()
        text_id += batch_size
    return text_feat

def get_lseg_scores(
        clip_modle,
        landmarks,
        lseg_map,
        clip_dim,
        add_other=True
) :
    '''
    Args:
    clip_modle: CLIP text encoder
    landmarks: list of landmark names
    lseg_map: numpy array of lseg map. shape: (H, W, clip_dim)
    clip_dim: dimension of CLIP text feature
    add_other: whether to add "other" as a landmark
    Returns:
    lseg_scores: numpy array of lseg scores. shape: (H, W, len(landmarks))
    '''
    landmarks_other = landmarks.copy()
    if add_other:
        landmarks_other.append("other")
    text_feat = get_text_feat(clip_modle, landmarks_other, clip_dim)
    map_feat = lseg_map.reshape(-1, lseg_map.shape[-1])
    lseg_score = lseg_map@text_feat.T

    return lseg_score

def create_lseg_map(
    data_dir,
    clip_version,
    clip_feat_dim,
    checkpoint_dir,
    camera_height,
    gs=1000,
    cs=0.5,
    crop_size=480,
    base_size=520,
    depth_sample_rate=100,
    cam_mat=None,
    fov=90,
    min_depth=0.1,
    max_depth=10,
):
    '''
    Generates a top-down semantic map using LSeg features from an RGB-D dataset with pose information.

    This function loads synchronized RGB images, depth maps, and camera poses from the dataset directory.
    It uses the LSeg model to extract pixel-wise CLIP-aligned semantic features, back-projects valid depth
    points into 3D space, and accumulates the features into a fixed-resolution 2D top-down grid map.

    Args:
        data_dir (str): Path to a directory containing 'color/', 'depth/', and 'pose/' subfolders.
        clip_version (str): CLIP model variant used by LSeg (e.g., 'ViT-B/32').
        clip_feat_dim (int): Dimensionality of the CLIP-aligned features.
        checkpoint_dir (str): Path to the pretrained LSeg model checkpoint.
        camera_height (float): Height of the camera from the ground in meters.
        gs (int): Grid size (number of cells per axis in the output top-down map).
        cs (float): Cell size in meters (world units per grid cell).
        crop_size (int): Crop size used when running LSeg on input RGB images.
        base_size (int): Resize base used by LSeg preprocessing pipeline.
        depth_sample_rate (int): Downsampling rate for pixels in the depth map (1 = dense, 100 = sparse).
        cam_mat (np.ndarray): Optional 3×3 camera intrinsic matrix. If None, computed from FOV.
        fov (float): Horizontal field of view in degrees (used only if cam_mat is None).
        min_depth (float): Minimum valid depth value in meters.
        max_depth (float): Maximum valid depth value in meters.

    Returns:
        color_top_down (np.ndarray): (gs, gs, 3) RGB visualization of the top-down semantic map.
        grid (np.ndarray): (gs, gs, clip_feat_dim) grid of aggregated CLIP features per cell.
        obstacles (np.ndarray): (gs, gs) binary occupancy map indicating observed obstacles.
    '''

    # Load file paths
    rgb_list, depth_list, pose_list = load_scene_path(data_dir)

    # Load LSeg model
    lseg_model = load_lseg_model(checkpoint_dir, labels=['dummy'], crop_size=crop_size).to(device)
    lseg_model.eval()

    # Load CLIP model
    clip_model, preprocess = clip.load(clip_version, device=device)
    clip_model.eval()

    # Prepare transform for RGB images
    norm_mean, norm_std = [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(norm_mean, norm_std),
    ])

    # Initialize top-down map containers
    color_top_down_height = (camera_height + 1) * np.ones((gs, gs), dtype=np.float32)
    color_top_down = np.zeros((gs, gs, 3), dtype=np.uint8)
    grid = np.zeros((gs, gs, clip_feat_dim), dtype=np.float32)
    obstacles = np.ones((gs, gs), dtype=np.uint8)
    weight = np.zeros((gs, gs), dtype=float)

    tf_list = []
    pbar = tqdm(range(len(rgb_list)), desc="Creating LSeg Map")
    count=0
    # Process each frame
    for rgb_path, depth_path, pose_path in zip(rgb_list, depth_list, pose_list):
        # Load RGB image
        bgr = cv2.imread(rgb_path)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        # Load depth map
        depth = load_depth(depth_path)*0.001

        # Load pose (robot to world)
        rot ,pos = load_pose(pose_path)

        # Convert robot pose to camera pose
        pos[2] += camera_height

        pose = np.eye(4)
        pose[:3, :3] = rot
        pose[:3, 3] = pos.flatten()

        # Align poses
        tf_list.append(pose)
        if len(tf_list) == 1:
            init_tf_inv = np.linalg.inv(tf_list[0])
        tf = init_tf_inv @ pose  # Local pose relative to the first frame

        # Extract LSeg pixel-wise feature map
        lseg_feat = get_lseg_feat(
            model=lseg_model,
            image=rgb,
            labels=['dummy'],
            transform=transform,
            crop_size=crop_size,
            base_size=base_size,
            norm_mean=norm_mean,
            norm_std=norm_std,
        )

        # Generate point cloud from depth
        pc, mask = depth2pc(depth,cam_mat,min_depth=min_depth,max_depth=max_depth)

        # Subsample points
        sample_indices = np.arange(pc.shape[1])
        np.random.shuffle(sample_indices)
        sample_indices = sample_indices[::depth_sample_rate]

        pc = pc[:, sample_indices]
        mask = mask[sample_indices]
        pc = pc[:, mask]  # Apply valid mask


        # Transform to global frame
        pc_global = transform_pc(pc, tf)


        # Project points into map
        for p_global, p_local in zip(pc_global.T, pc.T):

            x, y = pos2grid_id(gs, cs, -p_global[1], p_global[0])

            # Skip points outside the grid or from ceiling
            if x < 0 or y < 0 or x >= gs or y >= gs or p_local[2] > 2.0:
                continue

            # Project to RGB image to get color
            rgb_px, rgb_py, _ = project_point(cam_mat, p_local)
            if 0 <= rgb_px < rgb.shape[1] and 0 <= rgb_py < rgb.shape[0]:
                rgb_value = rgb[rgb_py, rgb_px, :]

                # Update color top-down map if closer
                if p_local[2] > color_top_down_height[y, x]:
                    color_top_down[y, x] = rgb_value
                    color_top_down_height[y, x] = p_local[2]

            # Project to feature map to get visual features
            feat_px, feat_py, _ = project_point(cam_mat, p_local)
            if 0 <= feat_px < lseg_feat.shape[3] and 0 <= feat_py < lseg_feat.shape[2]:
                feat = lseg_feat[0, :, feat_py, feat_px]
                grid[y, x] = (grid[y, x] * weight[y, x] + feat) / (weight[y, x] + 1)
                weight[y, x] += 1

            # Update obstacle map (0 means occupied)
            if p_local[2] >= camera_height:
                obstacles[y, x] = 0

        pbar.update(1)
        count+=1



    pbar.close()

    return color_top_down, grid, obstacles

clip_version = "ViT-B/32"
clip_feat_dim = {'RN50': 1024, 'RN101': 512, 'RN50x4': 640, 'RN50x16': 768,
                'RN50x64': 1024, 'ViT-B/32': 512, 'ViT-B/16': 512, 'ViT-L/14': 768}[clip_version]
clip_model, preprocess = clip.load(clip_version)  # clip.available_models()
clip_model.to(device).eval()

data_path='./data/'

cam_mat = np.array([
      [614.7241, 0,       319.6286],
      [0,        614.9275, 241.2029],
      [0,        0,         1     ]
  ])

color_top_down,lseg_map,obstacles=create_lseg_map(
    data_dir=data_path,
    clip_version=clip_version,
    clip_feat_dim=clip_feat_dim,
    checkpoint_dir=checkpoint_dir,
    camera_height=0.2,
    gs=2000,
    cs=0.1,
    crop_size=480,
    base_size=520,
    depth_sample_rate=50,
    cam_mat=cam_mat,
    min_depth=0.3,
    max_depth=3
)

x_indices, y_indices = np.where(obstacles == 0)

xmin = np.min(x_indices)
xmax = np.max(x_indices)
ymin = np.min(y_indices)
ymax = np.max(y_indices)

print(np.unique(obstacles))
obstacles_pil = Image.fromarray(obstacles[xmin:xmax+1, ymin:ymax+1])
plt.figure(figsize=(8, 6), dpi=120)
plt.imshow(obstacles_pil, cmap='gray')
plt.show()

np.save('./maps/color_top_down.npy', color_top_down)
np.save('./maps/lseg_map.npy', lseg_map)
np.save('./maps/obstacles.npy', obstacles)