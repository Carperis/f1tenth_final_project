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
from PIL import Image

from utils import *
import clip
from lseg.modules.models.lseg_net import LSegEncNet

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

color_top_down_path = 'color_top_down.npy'
lseg_map_path = 'lseg_map.npy'
obstacles_path = 'obstacles.npy'

color_top_down = load_map(color_top_down_path)
lseg_map = load_map(lseg_map_path)
obstacles = load_map(obstacles_path)

x_indices, y_indices = np.where(obstacles == 0)

xmin = np.min(x_indices)
xmax = np.max(x_indices)
ymin = np.min(y_indices)
ymax = np.max(y_indices)

# ----> here you can define your classes
lang = [
    "void",
    "wall",
    "floor",
]

device = "mps"
clip_version = "ViT-B/32"
clip_feat_dim = {'RN50': 1024, 'RN101': 512, 'RN50x4': 640, 'RN50x16': 768,
                'RN50x64': 1024, 'ViT-B/32': 512, 'ViT-B/16': 512, 'ViT-L/14': 768}[clip_version]
clip_model, preprocess = clip.load(clip_version)  # clip.available_models()
clip_model.to(device).eval()

grid=lseg_map
no_map_mask = obstacles[xmin:xmax+1, ymin:ymax+1] > 0
obstacles_rgb = np.repeat(obstacles[xmin:xmax+1, ymin:ymax+1, None], 3, axis=2)
print(no_map_mask.shape)
# prompt= input()
# lang = prompt.split(",")
# lang = mp3dcat
text_feats = get_text_feat(clip_model,lang, clip_feat_dim)

map_feats = grid[xmin:xmax+1, ymin:ymax+1].reshape((-1, grid.shape[-1]))
scores_list = map_feats @ text_feats.T

predicts = np.argmax(scores_list, axis=1)
predicts = predicts.reshape((xmax-xmin+1, ymax-ymin+1))
floor_mask = predicts == 2

new_pallete = get_new_pallete(len(lang))
mask, patches = get_new_mask_pallete(predicts, new_pallete, out_label_flag=True, labels=lang)
seg = mask.convert("RGBA")
seg = np.array(seg)
seg[no_map_mask] = [225, 225, 225, 255]
seg[floor_mask] = [225, 225, 225, 255]
seg = Image.fromarray(seg)
plt.figure(figsize=(10, 6), dpi=120)
plt.legend(handles=patches, loc='upper left', bbox_to_anchor=(1., 1), prop={'size': 10})
plt.axis('off')
plt.title("VLMaps")
plt.imshow(seg)
plt.show()