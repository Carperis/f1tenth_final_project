import requests
import numpy as np
from typing import List, Tuple, Optional, Dict
from utils import grid2map_coords, map2grid_coords, map2px_coords, px2map_coords
import imageio.v2 as imageio  # Changed import for ImageIO v3 compatibility
from sklearn.cluster import DBSCAN  # Removed KMeans, kept DBSCAN

class MapInferenceClient:
    def __init__(self, server_url: str = "http://127.0.0.1:1234/infer", obstacle_map_file: str = None):
        self.server_url = server_url
        self.obstacle_map_file = obstacle_map_file
        self.obstacle_map = self._load_obstacle_map() if obstacle_map_file else None

    def _load_obstacle_map(self) -> Optional[np.ndarray]:
        loaded_map = imageio.imread(self.obstacle_map_file)
        return loaded_map == 0

    def _query_location_score_map(self, location_name: str) -> Optional[dict]:
        response = requests.get(self.server_url, params={"location_name": location_name})
        return response.json()

    def find_obj(
        self,
        obj_name: str,
        ref_pos: Optional[List[float]] = None,
        radius: Optional[float] = None,
        top_k: Optional[int] = None,
        score_thres: float = 0.0,
        mask: bool = True,
        cluster: bool = False,
        dist_thres: Optional[float] = None
    ) -> Tuple[List[List[float]], List[float]]:
        map_data = self._query_location_score_map(obj_name)

        if not map_data or "score_map" not in map_data or map_data["score_map"] is None:
            return [], []
        
        score_map = np.array(map_data["score_map"])

        if score_map.ndim != 2:
            return [], []

        candidate_indices_rc_all = np.argwhere(score_map >= score_thres)
        
        if candidate_indices_rc_all.shape[0] == 0:
            return [], []

        coords_rc_to_process = candidate_indices_rc_all

        if mask and self.obstacle_map is not None:
            map_coords_xy_all_list = grid2map_coords(candidate_indices_rc_all.tolist()) 
            if not map_coords_xy_all_list:
                coords_rc_to_process = np.empty((0,2), dtype=int)
            else:
                map_coords_xy_all = np.array(map_coords_xy_all_list)

                px_coords_raw_all_list = map2px_coords(map_coords_xy_all.tolist()) 
                if not px_coords_raw_all_list:
                    coords_rc_to_process = np.empty((0,2), dtype=int)
                else:
                    px_coords_raw_all = np.array(px_coords_raw_all_list)
                    pgm_c_all = np.round(px_coords_raw_all[:, 0]).astype(int)
                    pgm_r_all = np.round(px_coords_raw_all[:, 1]).astype(int)

                    valid_r_mask = (pgm_r_all >= 0) & (pgm_r_all < self.obstacle_map.shape[0])
                    valid_c_mask = (pgm_c_all >= 0) & (pgm_c_all < self.obstacle_map.shape[1])
                    bounds_mask = valid_r_mask & valid_c_mask

                    final_keep_mask = np.zeros(candidate_indices_rc_all.shape[0], dtype=bool)
                    
                    indices_within_bounds = np.where(bounds_mask)[0]

                    if indices_within_bounds.size > 0:
                        bounded_pgm_r = pgm_r_all[indices_within_bounds]
                        bounded_pgm_c = pgm_c_all[indices_within_bounds]
                        
                        is_free_at_bounded_coords = ~self.obstacle_map[bounded_pgm_r, bounded_pgm_c]
                        
                        final_keep_mask[indices_within_bounds] = is_free_at_bounded_coords
                    
                    coords_rc_to_process = candidate_indices_rc_all[final_keep_mask]
        
        if coords_rc_to_process.shape[0] == 0:
            return [], []

        scores_for_processed = score_map[coords_rc_to_process[:, 0], 
                                         coords_rc_to_process[:, 1]]

        if ref_pos and radius is not None and radius >= 0:
            current_coords_rc_list = map2grid_coords([list(ref_pos)]) 
            if current_coords_rc_list:
                current_coords_rc = current_coords_rc_list[0]
                curr_r, curr_c = int(current_coords_rc[0]), int(current_coords_rc[1])
                
                score_map_grid_resolution = 0.1
                grid_radius_sq = (radius / score_map_grid_resolution)**2 

                diff_r = coords_rc_to_process[:, 0] - curr_r
                diff_c = coords_rc_to_process[:, 1] - curr_c
                distances_sq = diff_r**2 + diff_c**2
                
                in_radius_mask = distances_sq <= grid_radius_sq
                
                coords_rc_final_candidates = coords_rc_to_process[in_radius_mask]
                scores_final_candidates = scores_for_processed[in_radius_mask]
            else: 
                coords_rc_final_candidates = coords_rc_to_process
                scores_final_candidates = scores_for_processed
        else: 
            coords_rc_final_candidates = coords_rc_to_process
            scores_final_candidates = scores_for_processed

        if coords_rc_final_candidates.shape[0] == 0:
            return [], []

        result_coords_rc_array = np.empty((0,2))
        result_scores_array = np.array([])

        if cluster:
            if dist_thres is not None and dist_thres > 0:
                # --- DBSCAN LOGIC ---
                score_map_grid_resolution = 0.1  # Assuming 0.1 m/grid cell resolution
                eps_grid = dist_thres / score_map_grid_resolution
                
                dbscan = DBSCAN(eps=eps_grid, min_samples=1) 
                cluster_labels = dbscan.fit_predict(coords_rc_final_candidates)

                unique_labels = np.unique(cluster_labels)
                
                if len(unique_labels) == 0:
                    pass 
                else:
                    cluster_centroids_rc_list = []
                    cluster_avg_scores_list = []
                    for label in unique_labels:
                        points_in_cluster_mask = (cluster_labels == label)
                        coords_in_cluster = coords_rc_final_candidates[points_in_cluster_mask]
                        scores_in_cluster = scores_final_candidates[points_in_cluster_mask]

                        if coords_in_cluster.shape[0] > 0:
                            centroid_rc = np.mean(coords_in_cluster, axis=0)
                            avg_score = np.mean(scores_in_cluster)
                            cluster_centroids_rc_list.append(centroid_rc)
                            cluster_avg_scores_list.append(avg_score)
                    
                    if cluster_centroids_rc_list:
                        dbscan_coords_rc = np.array(cluster_centroids_rc_list)
                        dbscan_scores = np.array(cluster_avg_scores_list)

                        sort_indices_dbscan = np.argsort(dbscan_scores)[::-1]
                        sorted_coords_rc_dbscan = dbscan_coords_rc[sort_indices_dbscan]
                        sorted_scores_dbscan = dbscan_scores[sort_indices_dbscan]
                        
                        if top_k is not None and top_k > 0:
                            num_to_take = min(top_k, sorted_coords_rc_dbscan.shape[0])
                            result_coords_rc_array = sorted_coords_rc_dbscan[:num_to_take]
                            result_scores_array = sorted_scores_dbscan[:num_to_take]
                        else:
                            result_coords_rc_array = sorted_coords_rc_dbscan
                            result_scores_array = sorted_scores_dbscan
            # If dist_thres is not valid for DBSCAN, or cluster is True but no dist_thres,
            # fall back to sorting individual points if top_k is specified, or all points.
            else: 
                sort_indices = np.argsort(scores_final_candidates)[::-1]
                if top_k is not None and top_k > 0:
                    num_to_take = min(top_k, len(sort_indices))
                    final_sorted_indices = sort_indices[:num_to_take]
                else:
                    final_sorted_indices = sort_indices
                
                if len(final_sorted_indices) > 0:
                    result_coords_rc_array = coords_rc_final_candidates[final_sorted_indices]
                    result_scores_array = scores_final_candidates[final_sorted_indices]
        else: # cluster is False -> sort individual points
            sort_indices = np.argsort(scores_final_candidates)[::-1]
            if top_k is not None and top_k > 0:
                num_to_take = min(top_k, len(sort_indices))
                final_sorted_indices = sort_indices[:num_to_take]
            else:
                final_sorted_indices = sort_indices
            
            if len(final_sorted_indices) > 0:
                result_coords_rc_array = coords_rc_final_candidates[final_sorted_indices]
                result_scores_array = scores_final_candidates[final_sorted_indices]
        
        if result_coords_rc_array.shape[0] == 0:
            return [], []
            
        result_coords_xy_list = grid2map_coords(result_coords_rc_array.tolist())
        return result_coords_xy_list, result_scores_array.tolist()

    def find_multi_objs(
        self,
        obj_names: List[str],
        ref_pos: Optional[List[float]] = None,
        radius: Optional[float] = None,
        top_k: Optional[int] = None,
        score_thres: float = 0.0,
        mask: bool = True,
        cluster: bool = False, # Added cluster parameter
        dist_thres: Optional[float] = None # Added dist_thres parameter
    ) -> Dict[str, Tuple[List[List[float]], List[float]]]:
        results: Dict[str, Tuple[List[List[float]], List[float]]] = {}
        for obj_name in obj_names:
            results[obj_name] = self.find_obj(
                obj_name,
                ref_pos,
                radius,
                top_k,
                score_thres,
                mask,
                cluster,
                dist_thres
            )
        return results

if __name__ == '__main__':
    from viz_map_points import PointVisualizer
    client = MapInferenceClient(
        server_url="http://127.0.0.1:1234/infer",
        obstacle_map_file="./maps/map.pgm"
    )
    
    # obj_name = "table"
    # ref_pos = [6, 3]
    # coords, scores = client.find_obj(
    #     obj_name, 
    #     top_k=20,
    #     mask=True,
    #     ref_pos=ref_pos,
    #     radius=5.0,
    #     cluster=True,
    #     dist_thres=2.0,
    #     score_thres=0.95
    # )
    # print(f"Found {len(coords)} coordinates for {obj_name} with scores: {scores}")
    # visualizer = PointVisualizer(map_file="./maps/map.pgm")
    # coords += [ref_pos]
    # visualizer.visualize_points(coords, point_label=obj_name, point_type="map")
    
    # Example using find_multi_objs
    multi_obj_names = ["table"]
    ref_pos = [6, 3]
    multi_results = client.find_multi_objs(
        multi_obj_names,
        ref_pos=ref_pos,
        radius=5.0,
        top_k=10,
        mask=True,
        cluster=True,
        dist_thres=None,
        score_thres=0.9
    )
    
    # Visualize all objects with different labels
    multi_visualizer = PointVisualizer(map_file="./maps/map.pgm")
    all_points = [ref_pos]  # Start with reference position
    
    for obj_name, (obj_coords, obj_scores) in multi_results.items():
        print(f"Found {len(obj_coords)} coordinates for {obj_name} with scores: {obj_scores}")
        if obj_coords:
            all_points.extend(obj_coords)
            # Visualize each object separately
            multi_visualizer.visualize_points(obj_coords + [ref_pos], point_label=obj_name, point_type="map")
    
    # Visualize all objects together
    multi_visualizer.visualize_points(all_points, point_label="all objects", point_type="map")