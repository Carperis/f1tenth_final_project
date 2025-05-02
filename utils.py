
import torch.nn.functional as F
from scipy.spatial.transform import Rotation as R
import pandas as pd
import numpy as np

import matplotlib.patches as mpatches


def load_pose(pose_file, pitch_deg):
    '''
    pose_file: CSV file, each row = [x, y, z, qx, qy, qz, qw]
    '''
    # Read CSV
    df = pd.read_csv(pose_file, header=None)  # <-- no header
    row = df.iloc[0].values  # first pose

    pos = np.array(row[:3], dtype=np.float32).reshape(3, 1)
    quat = row[3:]
    r = R.from_quat(quat)
    rot = r.as_matrix()

    # Apply inverse pitch correction (rotation around y-axis)
    pitch_rad = -np.deg2rad(pitch_deg)
    R_pitch_inv = np.array([
        [np.cos(pitch_rad), 0, np.sin(pitch_rad)],
        [0, 1, 0],
        [-np.sin(pitch_rad), 0, np.cos(pitch_rad)]
    ])
    rot = R_pitch_inv @ rot

    return rot, pos


def load_depth(depth_filepath):
    with open(depth_filepath, "rb") as f:
        depth = np.load(f)
    return depth


def pad_image(img, mean, std, crop_size):
    b, c, h, w = img.shape  # .size()
    assert c == 3
    padh = crop_size - h if h < crop_size else 0
    padw = crop_size - w if w < crop_size else 0
    pad_values = -np.array(mean) / np.array(std)
    img_pad = img.new().resize_(b, c, h + padh, w + padw)
    for i in range(c):
        # note that pytorch pad params is in reversed orders
        img_pad[:, i, :, :] = F.pad(img[:, i, :, :], (0, padw, 0, padh), value=pad_values[i])
    assert img_pad.size(2) >= crop_size and img_pad.size(3) >= crop_size
    return img_pad

def resize_image(img, h, w, **up_kwargs):
    return F.interpolate(img, (h, w), **up_kwargs)


def crop_image(img, h0, h1, w0, w1):
    return img[:, :, h0:h1, w0:w1]

def get_cam_mat_fov(h,w,fov):
    cam_mat = np.eye(3)
    cam_mat[0, 0] = cam_mat[1, 1] = w / (2.0 * np.tan(np.deg2rad(fov / 2)))
    cam_mat[0, 2] = w / 2.0
    cam_mat[1, 2] = h / 2.0
    return cam_mat

def get_cam_mat(h,w):
    cam_mat=np.eye(3)
    cam_mat[0,0]=cam_mat[1,1]=w/2
    cam_mat[0,2]=w/2
    cam_mat[1,2]=h/2
    return cam_mat


def depth2pc(depth, cam_mat=None, fov=90, min_depth=0.1, max_depth=10.0):
    h, w = depth.shape

    if cam_mat is None:
        cam_mat = get_cam_mat_fov(h, w, fov)
    cam_mat_inv = np.linalg.inv(cam_mat)

    y, x = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
    x = x.reshape((1, -1))
    y = y.reshape((1, -1))
    depth = depth.reshape((1, -1))

    p_2d = np.vstack([x, y, np.ones_like(x)])
    pc = cam_mat_inv @ p_2d
    pc = pc * depth

    # Convert to x-forward, y-left, z-up
    pc_converted = np.vstack([
        pc[2, :],       # x = z
        -pc[0, :],      # y = -x
        -pc[1, :]       # z = -y
    ])

    # x is forward → use it as depth
    depth_vals = pc_converted[0, :]
    mask = (depth_vals > min_depth) & (depth_vals < max_depth)

    return pc_converted, mask



def transform_pc(pc,pose):
    pc_homo = np.vstack([pc, np.ones((1,pc.shape[1]))])
    pc_global_homo = pose @ pc_homo
    pc_global = pc_global_homo[:3, :]
    return pc_global
    


def pos2grid_id(gs,cs,xx,yy):
    x = int(gs / 2 + int(xx / cs))
    y = int(gs / 2 - int(yy / cs))
    return [x, y]

def project_point(cam_mat, p_custom):
    """
    Project a 3D point in custom camera frame (x=forward, y=left, z=up)
    to image coordinates using camera intrinsics.

    Args:
        cam_mat (np.ndarray): 3x3 camera intrinsic matrix
        p_custom (np.ndarray): 3D point in (x=forward, y=left, z=up)

    Returns:
        (x_img, y_img, z_cam): pixel coordinates and depth (z in OpenCV frame)
        or None if point is behind the camera
    """
    # Convert custom frame → OpenCV frame
    p_cam = np.array([
        -p_custom[1],  # x = -left → right
        -p_custom[2],  # y = -up   → down
         p_custom[0]   # z = forward (same)
    ]).reshape((3, 1))

    # Project with intrinsics
    new_p = cam_mat @ p_cam
    z = new_p[2, 0]  # depth

    if z <= 0:
        return None  # point behind the camera

    new_p /= z
    x_img = int(new_p[0, 0] + 0.5)
    y_img = int(new_p[1, 0] + 0.5)

    return x_img, y_img, z

def get_new_pallete(num_cls):
    n = num_cls
    pallete = [0] * (n * 3)

    for j in range(0, n):
        lab = j
        pallete[j * 3 + 0] = 0
        pallete[j * 3 + 1] = 0
        pallete[j * 3 + 2] = 0
        i = 0
        while lab > 0:
            pallete[j * 3 + 0] |= ((lab >> 0) & 1) << (7 - i)
            pallete[j * 3 + 1] |= ((lab >> 1) & 1) << (7 - i)
            pallete[j * 3 + 2] |= ((lab >> 2) & 1) << (7 - i)
            i = i + 1
            lab >>= 3
    return pallete


def get_new_mask_pallete(npimg, new_palette, out_label_flag=False, labels=None, ignore_ids_list=[]):
    """Get image color pallete for visualizing masks"""
    # put colormap
    out_img = Image.fromarray(npimg.squeeze().astype("uint8"))
    out_img.putpalette(new_palette)

    if out_label_flag:
        assert labels is not None
        u_index = np.unique(npimg)
        patches = []
        for i, index in enumerate(u_index):
            if index in ignore_ids_list:
                continue
            label = labels[index]
            cur_color = [
                new_palette[index * 3] / 255.0,
                new_palette[index * 3 + 1] / 255.0,
                new_palette[index * 3 + 2] / 255.0,
            ]
            red_patch = mpatches.Patch(color=cur_color, label=label)
            patches.append(red_patch)
    return out_img, patches


def load_map(load_path):
    with open(load_path, "rb") as f:
        map = np.load(f)
    return map