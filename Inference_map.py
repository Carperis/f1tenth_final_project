import sys
import os
import imageio
import numpy as np
import cv2
import tqdm
from IPython.display import HTML
from base64 import b64encode
import torch
import math
from tqdm import tqdm
from torchvision.transforms import transforms
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw

from utils import *
import clip
from lseg.modules.models.lseg_net import LSegEncNet

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")


def get_text_feat(clip_model, in_text, clip_dim, batch_size=64):
    """
    Args:
    CLIP text encoder
    in_text: list of texts
    clip_dim: dimension of CLIP text feature
    batch_size: batch size for encoding
    Returns:
    text_feat: numpy array of text features. shape: (len(in_text), clip_dim)
    """
    if torch.cuda.is_available():
        text_tokens = clip.tokenize(in_text).cuda()
    elif torch.backends.mps.is_available():
        text_tokens = clip.tokenize(in_text).to("mps")
    else:
        text_tokens = clip.tokenize(in_text)

    text_id = 0
    text_feat = np.zeros((len(in_text), clip_dim), dtype=np.float32)

    while text_id < len(in_text):
        batch_size = min(len(in_text) - text_id, batch_size)
        text_tokens_batch = text_tokens[text_id : text_id + batch_size]
        with torch.no_grad():
            batch_feats = clip_model.encode_text(text_tokens_batch).float()
        batch_feats /= batch_feats.norm(dim=-1, keepdim=True)
        text_feat[text_id : text_id + batch_size] = batch_feats.cpu().numpy()
        text_id += batch_size
    return text_feat


def find_top_k_points(prompt_text, k, lang_list, all_scores, original_predicts_shape, base_image):
    """
    Finds the top k score points for a given prompt and visualizes them on the image.

    Args:
        prompt_text (str): The text prompt (e.g., "door").
        k (int): The number of top points to find.
        lang_list (list): List of all class labels.
        all_scores (np.ndarray): Array of scores for each point against each label. Shape (num_points, num_labels).
        original_predicts_shape (tuple): The shape of the predicts map (rows, cols).
        base_image (PIL.Image): The base image to draw on.

    Returns:
        list: A list of (row, col) tuples for the top k points.
    """
    try:
        prompt_idx = lang_list.index(prompt_text)
    except ValueError:
        print(f"Prompt '{prompt_text}' not found in language list: {lang_list}")
        return []

    # Scores for the specific prompt
    prompt_scores = all_scores[:, prompt_idx]

    # Get indices of top k scores (flat indices)
    # These indices correspond to the flattened map_feats / scores_list
    top_k_flat_indices = np.argsort(prompt_scores)[-k:]

    # Convert flat indices to 2D coordinates
    # The original_predicts_shape is (rows, cols) which corresponds to (xmax-xmin+1, ymax-ymin+1)
    rows, cols = np.unravel_index(top_k_flat_indices, original_predicts_shape)

    top_k_coords = list(zip(rows.tolist(), cols.tolist()))
    
    # For scatter plot, x-coordinates are columns, y-coordinates are rows
    plot_cols = [c for r, c in top_k_coords]
    plot_rows = [r for r, c in top_k_coords]

    plt.figure(figsize=(10, 7), dpi=120)
    plt.imshow(base_image)
    plt.scatter(plot_cols, plot_rows, c='red', s=40, marker='o', edgecolors='black') # s is size, marker can be changed
    plt.title(f"Top {k} points for '{prompt_text}'")
    plt.axis("off")
    # plt.show() # Removed to allow further plotting or showing later

    # Visualize the points
    # img_with_points = base_image.copy()
    # draw = ImageDraw.Draw(img_with_points)
    # radius = 5  # Radius of the circle marker
    # for r, c in top_k_coords:
    #     # Draw a circle. Note: PIL's draw coordinates are (x,y) which is (col,row)
    #     draw.ellipse([(c - radius, r - radius), (c + radius, r + radius)], fill="red", outline="red")

    # plt.figure(figsize=(10, 7), dpi=120)
    # plt.imshow(img_with_points)
    # plt.title(f"Top {k} points for '{prompt_text}'")
    # plt.axis("off")
    # plt.show()

    return top_k_coords


color_top_down_path = "color_top_down.npy"
lseg_map_path = "lseg_map.npy"
obstacles_path = "obstacles.npy"

color_top_down = load_map(color_top_down_path)
lseg_map = load_map(lseg_map_path)
obstacles = load_map(obstacles_path)

x_indices, y_indices = np.where(obstacles == 0)

# xmin = np.min(x_indices)
# xmax = np.max(x_indices)
# ymin = np.min(y_indices)
# ymax = np.max(y_indices)

xmin = 0
xmax = lseg_map.shape[0] - 1
ymin = 0
ymax = lseg_map.shape[1] - 1

# ----> here you can define your classes
lang = [
    "void",
    "wall",
    "floor",
    "chair",
    "table",
    "door",
]

clip_version = "ViT-B/32"
clip_feat_dim = {"RN50": 1024, "RN101": 512, "RN50x4": 640, "RN50x16": 768, "RN50x64": 1024, "ViT-B/32": 512, "ViT-B/16": 512, "ViT-L/14": 768}[clip_version]
clip_model, preprocess = clip.load(clip_version)  # clip.available_models()
clip_model.to(device).eval()

grid = lseg_map
no_map_mask = obstacles[xmin : xmax + 1, ymin : ymax + 1] > 0
obstacles_rgb = np.repeat(obstacles[xmin : xmax + 1, ymin : ymax + 1, None], 3, axis=2)
print(no_map_mask.shape)
# prompt= input()
# lang = prompt.split(",")
# lang = mp3dcat
text_feats = get_text_feat(clip_model, lang, clip_feat_dim)

map_feats = grid[xmin : xmax + 1, ymin : ymax + 1].reshape((-1, grid.shape[-1]))
scores_list = map_feats @ text_feats.T
print("scores_list shape", scores_list.shape)

predicts = np.argmax(scores_list, axis=1)
predicts = predicts.reshape((xmax - xmin + 1, ymax - ymin + 1))
floor_mask = predicts == 2

new_pallete = get_new_pallete(len(lang))
mask, patches = get_new_mask_pallete(predicts, new_pallete, out_label_flag=True, labels=lang)
seg = mask.convert("RGBA")
seg = np.array(seg)
seg[no_map_mask] = [225, 225, 225, 255]
seg[floor_mask] = [225, 225, 225, 255]
seg = Image.fromarray(seg)
print("seg shape", seg.size)
plt.figure(figsize=(10, 6), dpi=120)
plt.legend(handles=patches, loc="upper left", bbox_to_anchor=(1.0, 1), prop={"size": 10})
plt.axis("off")
plt.title("VLMaps")
plt.imshow(seg)

# Example usage of the new function:
# Ensure 'door' is in your 'lang' list or change the prompt
user_prompt = "chair"
top_n = 5
if user_prompt in lang:
    top_points = find_top_k_points(
        prompt_text=user_prompt,
        k=top_n,
        lang_list=lang,
        all_scores=scores_list,  # This is map_feats @ text_feats.T
        original_predicts_shape=(xmax - xmin + 1, ymax - ymin + 1),  # Shape of 'predicts'
        base_image=seg,  # The segmented image generated earlier
    )
    if top_points:
        print(f"Top {top_n} coordinates for '{user_prompt}': {top_points}")
else:
    print(f"Prompt '{user_prompt}' is not in the defined language classes: {lang}")

plt.show()
