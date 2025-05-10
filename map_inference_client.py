import requests
import numpy as np
from typing import List, Tuple, Optional, Dict
from utils import grid2map_coords, map2grid_coords, map2px_coords, px2map_coords
import imageio.v2 as imageio  # Changed import for ImageIO v3 compatibility

class MapInferenceClient:
    def __init__(self, server_url: str = "http://127.0.0.1:1234/infer", obstacle_map_file: str = "maps/map.pgm"):
        self.server_url = server_url
        self.obstacle_map_pgm_path = obstacle_map_file
        self.obstacle_map = self._load_obstacle_map()

    def _load_obstacle_map(self) -> Optional[np.ndarray]:
        try:
            loaded_map = imageio.imread(self.obstacle_map_pgm_path)
            return loaded_map == 0
        except FileNotFoundError:
            print(f"Warning: Obstacle map PGM file not found at {self.obstacle_map_pgm_path}. Obstacle filtering will be skipped.")
            return None
        except Exception as e:
            print(f"Warning: Error loading obstacle map PGM: {e}. Obstacle filtering will be skipped.")
            return None

    def _query_location_score_map(self, location_name: str) -> Optional[dict]:
        response = requests.get(self.server_url, params={"location_name": location_name})
        response.raise_for_status()
        return response.json()

    def find_obj_coords(
        self,
        obj_name: str,
        current_coords_xy: Optional[Tuple[float, float]] = None,
        radius: Optional[float] = None,
        top_k: Optional[int] = None,
        score_thres: float = 0.0,
        filter: bool = True
    ) -> List[List[float]]:
        map_data = self._query_location_score_map(obj_name)

        if not map_data or "score_map" not in map_data or map_data["score_map"] is None:
            return []
        
        score_map = np.array(map_data["score_map"])

        if score_map.ndim != 2:
            return []

        candidate_indices_rc_all = np.argwhere(score_map >= score_thres) # Shape (N, 2) [r, c]
        
        if candidate_indices_rc_all.shape[0] == 0:
            return []

        coords_rc_to_process = candidate_indices_rc_all

        if filter and self.obstacle_map is not None:
            # Vectorized obstacle filtering
            # 1. Convert grid (r,c) to map (x,y) coordinates
            # grid2map_coords expects list of lists [r, c] or [gx, gy]
            map_coords_xy_all_list = grid2map_coords(candidate_indices_rc_all.tolist()) 
            if not map_coords_xy_all_list:
                coords_rc_to_process = np.empty((0,2), dtype=int)
            else:
                map_coords_xy_all = np.array(map_coords_xy_all_list)

                # 2. Convert map (x,y) to PGM pixel (px_x_raw, px_y_raw) coordinates
                px_coords_raw_all_list = map2px_coords(map_coords_xy_all.tolist()) 
                if not px_coords_raw_all_list:
                    coords_rc_to_process = np.empty((0,2), dtype=int)
                else:
                    px_coords_raw_all = np.array(px_coords_raw_all_list)
                    # 3. Round to integer pixel coordinates for indexing
                    # PGM pixel columns (pgm_c) from px_x_raw, PGM pixel rows (pgm_r) from px_y_raw
                    pgm_c_all = np.round(px_coords_raw_all[:, 0]).astype(int)
                    pgm_r_all = np.round(px_coords_raw_all[:, 1]).astype(int)

                    # 4. Bounds check for PGM map
                    valid_r_mask = (pgm_r_all >= 0) & (pgm_r_all < self.obstacle_map.shape[0])
                    valid_c_mask = (pgm_c_all >= 0) & (pgm_c_all < self.obstacle_map.shape[1])
                    bounds_mask = valid_r_mask & valid_c_mask

                    # 5. Obstacle check for points within bounds
                    final_keep_mask = np.zeros(candidate_indices_rc_all.shape[0], dtype=bool)
                    
                    indices_within_bounds = np.where(bounds_mask)[0]

                    if indices_within_bounds.size > 0:
                        bounded_pgm_r = pgm_r_all[indices_within_bounds]
                        bounded_pgm_c = pgm_c_all[indices_within_bounds]
                        
                        is_free_at_bounded_coords = ~self.obstacle_map[bounded_pgm_r, bounded_pgm_c]
                        
                        final_keep_mask[indices_within_bounds] = is_free_at_bounded_coords
                    
                    coords_rc_to_process = candidate_indices_rc_all[final_keep_mask]
        
        if coords_rc_to_process.shape[0] == 0:
            return []

        # Get scores for these processed candidates
        scores_for_processed = score_map[coords_rc_to_process[:, 0], 
                                         coords_rc_to_process[:, 1]]

        # Radius filtering
        if current_coords_xy and radius is not None and radius >= 0:
            # map2grid_coords expects list of lists, returns list of lists
            current_coords_cr_list = map2grid_coords([list(current_coords_xy)]) 
            if current_coords_cr_list:
                current_coords_cr = current_coords_cr_list[0] # [c, r]
                curr_r, curr_c = int(current_coords_cr[1]), int(current_coords_cr[0])
                
                score_map_grid_resolution = map_data.get("resolution", 0.1) 
                # Using squared distance to avoid sqrt
                grid_radius_sq = (radius / score_map_grid_resolution)**2 

                diff_r = coords_rc_to_process[:, 0] - curr_r
                diff_c = coords_rc_to_process[:, 1] - curr_c
                distances_sq = diff_r**2 + diff_c**2
                
                in_radius_mask = distances_sq <= grid_radius_sq
                
                coords_rc_final_candidates = coords_rc_to_process[in_radius_mask]
                scores_final_candidates = scores_for_processed[in_radius_mask]
            else: # current_coords_cr_list was empty (should not happen if current_coords_xy is valid)
                coords_rc_final_candidates = coords_rc_to_process
                scores_final_candidates = scores_for_processed
        else: # No radius filtering
            coords_rc_final_candidates = coords_rc_to_process
            scores_final_candidates = scores_for_processed

        if coords_rc_final_candidates.shape[0] == 0:
            return []

        # Sort by scores (descending)
        sort_indices = np.argsort(scores_final_candidates)[::-1] # Indices to sort in descending order
        
        # Apply top_k
        if top_k is not None and top_k > 0:
            num_to_take = min(top_k, len(sort_indices))
            final_sorted_indices = sort_indices[:num_to_take]
        else:
            final_sorted_indices = sort_indices
            
        result_coords_rc_array = coords_rc_final_candidates[final_sorted_indices]
        
        if result_coords_rc_array.shape[0] == 0:
            return []
            
        # Convert final [r,c] grid coordinates to [x,y] map coordinates
        # grid2map_coords expects list of lists [gx, gy] which we treat as [r, c]
        result_coords_xy_list = grid2map_coords(result_coords_rc_array.tolist())
        return result_coords_xy_list

    def find_multi_obj_coords(
        self,
        obj_names: List[str],
        current_coords_xy: Optional[Tuple[float, float]] = None,
        radius: Optional[float] = None,
        top_k: Optional[int] = None,
        score_thres: float = 0.0,
        filter: bool = True  # Added filter parameter
    ) -> Dict[str, List[List[float]]]:
        results: Dict[str, List[List[float]]] = {}
        for obj_name in obj_names:
            results[obj_name] = self.find_obj_coords(
                obj_name,
                current_coords_xy,
                radius,
                top_k,
                score_thres,
                filter  # Pass filter flag
            )
        return results

if __name__ == '__main__':
    client = MapInferenceClient()
    
    object_name = "chair"
    coords = client.find_obj_coords(object_name, top_k=10, filter=True)
    from viz_map_points import PointVisualizer
    visualizer = PointVisualizer(
        map_file="maps/map.pgm"
    )
    visualizer.visualize_points(coords, point_label=object_name, point_type="map")

    # print("--- Testing single object: 'door' (with filtering by default) ---")
    # door_coords = client.find_obj_coords("door", top_k=3, score_thres=0.5)
    # print(f"Found door coordinates: {door_coords}" if door_coords else "No door coordinates found.")

    # print("\n--- Testing single object: 'door' (explicitly with filtering) ---")
    # door_coords_filtered = client.find_obj_coords("door", top_k=3, score_thres=0.5, filter=True)
    # print(f"Found door coordinates (filtered): {door_coords_filtered}" if door_coords_filtered else "No door coordinates found (filtered).")

    # print("\n--- Testing single object: 'door' (without filtering) ---")
    # door_coords_unfiltered = client.find_obj_coords("door", top_k=3, score_thres=0.5, filter=False)
    # print(f"Found door coordinates (unfiltered): {door_coords_unfiltered}" if door_coords_unfiltered else "No door coordinates found (unfiltered).")


    # print("\n--- Testing single object: 'table' with regional search (with filtering by default) ---")
    # table_coords_regional = client.find_obj_coords(
    #     "table",
    #     current_coords_xy=(10.0, 5.0),
    #     radius=3.0,
    #     top_k=2,
    #     score_thres=0.4
    # )
    # print(f"Found table coordinates (regional): {table_coords_regional}" if table_coords_regional else "No table coordinates found (regional).")
    
    # print("\n--- Testing multiple objects: ['chair', 'window', 'nonexistent_object'] (with filtering by default) ---")
    # objects_to_find = ["chair", "window", "nonexistent_object"]
    # multi_object_results = client.find_multi_obj_coords(
    #     objects_to_find,
    #     top_k=2,
    #     score_thres=0.3
    # )
    # for obj_name, coords in multi_object_results.items():
    #     print(f"Coordinates for '{obj_name}': {coords}" if coords else f"No coordinates found for '{obj_name}'.")

    # print("\n--- Testing multiple objects: ['chair', 'window'] (without filtering) ---")
    # objects_to_find_no_filter = ["chair", "window"]
    # multi_object_results_no_filter = client.find_multi_obj_coords(
    #     objects_to_find_no_filter,
    #     top_k=2,
    #     score_thres=0.3,
    #     filter=False
    # )
    # for obj_name, coords in multi_object_results_no_filter.items():
    #     print(f"Coordinates for '{obj_name}' (unfiltered): {coords}" if coords else f"No coordinates found for '{obj_name}' (unfiltered).")

    # print("\nClient tests finished.")
