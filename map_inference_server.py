import os
import numpy as np
import torch
import clip
from flask import Flask, request, jsonify
from PIL import Image

# Assuming utils.py and LSeg model files are accessible
from utils import load_map # Make sure load_map is in utils.py or define it here
# from lseg.modules.models.lseg_net import LSegEncNet # If LSegEncNet is needed directly

# --- Configuration & Model Loading ---
# Determine device
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print(f"Using device: {device}")

# Paths to map files
LSEG_MAP_PATH = "lseg_map.npy"
OBSTACLES_MAP_PATH = "obstacles.npy"

# Load maps
print("Loading lseg_map...")
lseg_map = load_map(LSEG_MAP_PATH)
if lseg_map is None:
    raise FileNotFoundError(f"{LSEG_MAP_PATH} not found or failed to load.")
print(f"lseg_map loaded, shape: {lseg_map.shape}")

print("Loading obstacles_map...")
obstacles_map = load_map(OBSTACLES_MAP_PATH)
if obstacles_map is None:
    raise FileNotFoundError(f"{OBSTACLES_MAP_PATH} not found or failed to load.")
print(f"obstacles_map loaded, shape: {obstacles_map.shape}")

# Define map boundaries (adjust if necessary, similar to Inference_map.py)
# Assuming full map for now, can be refined with xmin, xmax, etc.
MAP_ROWS, MAP_COLS, MAP_FEAT_DIM = lseg_map.shape
original_predicts_shape = (MAP_ROWS, MAP_COLS)

# Flattened map features
# Ensure lseg_map is (H, W, C) and reshape to (H*W, C)
map_feats_flat = lseg_map.reshape(-1, MAP_FEAT_DIM)
print(f"Flattened map features shape: {map_feats_flat.shape}")

# --- CLIP Model ---
print("Loading CLIP model...")
CLIP_VERSION = "ViT-B/32"
try:
    clip_model, _ = clip.load(CLIP_VERSION, device=device)
    clip_model.eval()
    CLIP_FEAT_DIM = clip_model.text_projection.shape[-1] # Get dim from model
except Exception as e:
    print(f"Error loading CLIP model: {e}")
    raise

print(f"CLIP model {CLIP_VERSION} loaded. Feature dimension: {CLIP_FEAT_DIM}")

# --- Language Labels (Classes) ---
# These should match the classes your LSeg model was trained on or expects
# Or the classes you want to query for.
LANG_LABELS = [
    "void", "wall", "floor"
    # Add more relevant classes for your environment
]
print(f"Defined language labels: {LANG_LABELS}")

def get_text_features(model, text_list, model_dim):
    """Encodes a list of text strings into CLIP features."""
    tokens = clip.tokenize(text_list).to(device)
    with torch.no_grad():
        text_features = model.encode_text(tokens).float()
        text_features /= text_features.norm(dim=-1, keepdim=True)
    return text_features.cpu().numpy()

# Precompute text features for all language labels
print("Precomputing text features for language labels...")
try:
    text_features_all_labels = get_text_features(clip_model, LANG_LABELS, CLIP_FEAT_DIM)
    print(f"Text features computed, shape: {text_features_all_labels.shape}")
except Exception as e:
    print(f"Error computing text features: {e}")
    raise

# --- Flask App ---
app = Flask(__name__)

@app.route('/infer', methods=['GET'])
def infer_location():
    location_name = request.args.get('location_name')
    if not location_name:
        return jsonify({"error": "Missing 'location_name' parameter"}), 400

    try:
        # Compute text features for the requested location_name on the fly
        requested_text_features = get_text_features(clip_model, [location_name], CLIP_FEAT_DIM)
        
        # Squeeze to get (CLIP_FEAT_DIM,) for dot product
        requested_text_features_squeezed = requested_text_features.squeeze()

        if map_feats_flat.shape[1] != requested_text_features_squeezed.shape[0]:
            error_msg = (
                f"Dimension mismatch: Map features ({map_feats_flat.shape[1]}) vs "
                f"Text features ({requested_text_features_squeezed.shape[0]}) for '{location_name}'."
            )
            print(error_msg) # Log for server-side debugging
            return jsonify({"error": error_msg}), 500

        # Calculate scores for the specific label
        scores_for_label = np.dot(map_feats_flat, requested_text_features_squeezed) # (num_points,)

        # Reshape scores to map dimensions
        score_map = scores_for_label.reshape(original_predicts_shape)

        # Convert to list of lists for JSON serialization
        score_map_list = score_map.tolist()

        return jsonify({
            "location_name": location_name,
            "score_map_shape": original_predicts_shape,
            "score_map": score_map_list
        })

    except Exception as e:
        # Catch any other errors during feature computation or processing
        print(f"Error processing location '{location_name}': {e}") # Log to server console
        return jsonify({"error": f"Failed to process location '{location_name}'. Details: {str(e)}"}), 500

if __name__ == '__main__':
    # Make sure to run this from the directory where npy files and utils.py are,
    # or adjust paths accordingly.
    print("Starting Flask server...")
    app.run(host='0.0.0.0', port=1234, debug=True, use_reloader=False)
