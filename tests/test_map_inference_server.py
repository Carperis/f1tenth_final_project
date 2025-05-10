import os
import numpy as np
import torch
import clip
from flask import Flask, request, jsonify
from PIL import Image
from utils import load_map

# --- Configuration & Model Loading ---
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

LSEG_MAP_PATH = "./maps/lseg_map.npy"
OBSTACLES_MAP_PATH = "./maps/obstacles.npy"

lseg_map = load_map(LSEG_MAP_PATH)
if lseg_map is None:
    raise FileNotFoundError(f"{LSEG_MAP_PATH} not found or failed to load.")

obstacles_map = load_map(OBSTACLES_MAP_PATH)
if obstacles_map is None:
    raise FileNotFoundError(f"{OBSTACLES_MAP_PATH} not found or failed to load.")

MAP_ROWS, MAP_COLS, MAP_FEAT_DIM = lseg_map.shape
original_predicts_shape = (MAP_ROWS, MAP_COLS)
map_feats_flat = lseg_map.reshape(-1, MAP_FEAT_DIM)

# --- CLIP Model ---
CLIP_VERSION = "ViT-B/32"
clip_model, _ = clip.load(CLIP_VERSION, device=device)
clip_model.eval()
CLIP_FEAT_DIM = clip_model.text_projection.shape[-1]

# --- Language Labels (Classes) ---
LANG_LABELS = ["void", "wall", "floor"]

def get_text_features(model, text_list):
    """Encodes a list of text strings into CLIP features."""
    tokens = clip.tokenize(text_list).to(device)
    with torch.no_grad():
        text_features = model.encode_text(tokens).float()
        text_features /= text_features.norm(dim=-1, keepdim=True)
    return text_features.cpu().numpy()

text_features_all_labels = get_text_features(clip_model, LANG_LABELS)

# --- Flask App ---
app = Flask(__name__)

@app.route('/infer', methods=['GET'])
def infer_location():
    location_name = request.args.get('location_name')
    if not location_name:
        return jsonify({"error": "Missing 'location_name' parameter"}), 400

    requested_text_features = get_text_features(clip_model, [location_name])
    requested_text_features_squeezed = requested_text_features.squeeze()

    if map_feats_flat.shape[1] != requested_text_features_squeezed.shape[0]:
        error_msg = (
            f"Dimension mismatch: Map features ({map_feats_flat.shape[1]}) vs "
            f"Text features ({requested_text_features_squeezed.shape[0]}) for '{location_name}'."
        )
        return jsonify({"error": error_msg}), 500

    scores_for_label = np.dot(map_feats_flat, requested_text_features_squeezed)
    score_map = scores_for_label.reshape(original_predicts_shape)
    score_map_list = score_map.tolist()

    return jsonify({
        "location_name": location_name,
        "score_map_shape": original_predicts_shape,
        "score_map": score_map_list
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=1234, debug=False, use_reloader=False)
