import numpy as np
import torch
import clip
from flask import Flask, request, jsonify
from utils import load_map

class MapInferenceServer:
    def __init__(self, lseg_map_path="./maps/lseg_map.npy", obstacles_map_path="./maps/obstacles.npy", clip_version="ViT-B/32"):
        self.device = self._initialize_device()
        
        self.lseg_map = load_map(lseg_map_path)
        if self.lseg_map is None:
            raise FileNotFoundError(f"{lseg_map_path} not found or failed to load.")

        self.obstacles_map = load_map(obstacles_map_path)
        if self.obstacles_map is None:
            raise FileNotFoundError(f"{obstacles_map_path} not found or failed to load.")

        map_rows, map_cols, map_feat_dim = self.lseg_map.shape
        self.original_predicts_shape = (map_rows, map_cols)
        self.map_feats_flat = self.lseg_map.reshape(-1, map_feat_dim)

        self.clip_model, _ = clip.load(clip_version, device=self.device)
        self.clip_model.eval()
        # self.clip_feat_dim = self.clip_model.text_projection.shape[-1] # Not strictly needed for current logic

        self.app = Flask(__name__)
        self.app.add_url_rule('/infer', view_func=self.infer_location, methods=['GET'])

    def _initialize_device(self):
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.backends.mps.is_available(): # For Apple Silicon
            return torch.device("mps")
        return torch.device("cpu")

    def _get_text_features(self, text_list):
        tokens = clip.tokenize(text_list).to(self.device)
        with torch.no_grad():
            text_features = self.clip_model.encode_text(tokens).float()
            text_features /= text_features.norm(dim=-1, keepdim=True)
        return text_features.cpu().numpy()

    def infer_location(self):
        location_name = request.args.get('location_name')
        if not location_name:
            return jsonify({"error": "Missing 'location_name' parameter"}), 400

        requested_text_features = self._get_text_features([location_name])
        requested_text_features_squeezed = requested_text_features.squeeze()

        if self.map_feats_flat.shape[1] != requested_text_features_squeezed.shape[0]:
            error_msg = (
                f"Dimension mismatch: Map features ({self.map_feats_flat.shape[1]}) vs "
                f"Text features ({requested_text_features_squeezed.shape[0]}) for '{location_name}'."
            )
            return jsonify({"error": error_msg}), 500

        scores_for_label = np.dot(self.map_feats_flat, requested_text_features_squeezed)
        # Normalize scores to range [0, 1]
        scores_min = np.min(scores_for_label)
        scores_max = np.max(scores_for_label)
        if scores_max > scores_min:  # Avoid division by zero
            scores_for_label = (scores_for_label - scores_min) / (scores_max - scores_min)
        else:
            scores_for_label = np.zeros_like(scores_for_label)  # If all values are the same
        score_map = scores_for_label.reshape(self.original_predicts_shape)
        
        return jsonify({
            "location_name": location_name,
            "score_map_shape": self.original_predicts_shape,
            "score_map": score_map.tolist()
        })

    def run(self, host='0.0.0.0', port=1234, debug=False):
        self.app.run(host=host, port=port, debug=debug, use_reloader=False)

if __name__ == '__main__':
    server = MapInferenceServer()
    server.run()
