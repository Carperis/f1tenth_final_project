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
        loaded_map = imageio.imread(self.obstacle_map_pgm_path)
        return loaded_map == 0

    def _query_location_score_map(self, location_name: str) -> Optional[dict]:
        response = requests.get(self.server_url, params={"location_name": location_name})
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

        candidate_indices_rc_all = np.argwhere(score_map >= score_thres)
        
        if candidate_indices_rc_all.shape[0] == 0:
            return []

        coords_rc_to_process = candidate_indices_rc_all

        if filter and self.obstacle_map is not None:
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
            return []

        scores_for_processed = score_map[coords_rc_to_process[:, 0], 
                                         coords_rc_to_process[:, 1]]

        if current_coords_xy and radius is not None and radius >= 0:
            current_coords_cr_list = map2grid_coords([list(current_coords_xy)]) 
            if current_coords_cr_list:
                current_coords_cr = current_coords_cr_list[0]
                curr_r, curr_c = int(current_coords_cr[1]), int(current_coords_cr[0])
                
                score_map_grid_resolution = map_data.get("resolution", 0.1) 
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
            return []

        sort_indices = np.argsort(scores_final_candidates)[::-1] 
        
        if top_k is not None and top_k > 0:
            num_to_take = min(top_k, len(sort_indices))
            final_sorted_indices = sort_indices[:num_to_take]
        else:
            final_sorted_indices = sort_indices
            
        result_coords_rc_array = coords_rc_final_candidates[final_sorted_indices]
        
        if result_coords_rc_array.shape[0] == 0:
            return []
            
        result_coords_xy_list = grid2map_coords(result_coords_rc_array.tolist())
        return result_coords_xy_list

    def find_multi_obj_coords(
        self,
        obj_names: List[str],
        current_coords_xy: Optional[Tuple[float, float]] = None,
        radius: Optional[float] = None,
        top_k: Optional[int] = None,
        score_thres: float = 0.0,
        filter: bool = True
    ) -> Dict[str, List[List[float]]]:
        results: Dict[str, List[List[float]]] = {}
        for obj_name in obj_names:
            results[obj_name] = self.find_obj_coords(
                obj_name,
                current_coords_xy,
                radius,
                top_k,
                score_thres,
                filter
            )
        return results

if __name__ == '__main__':
    client = MapInferenceClient()
    
    chair_coords_viz = client.find_obj_coords("chair", top_k=10, filter=True)
    if chair_coords_viz:
        from viz_map_points import PointVisualizer
        visualizer = PointVisualizer(map_file="maps/map.pgm")
        visualizer.visualize_points(chair_coords_viz, point_label="chair", point_type="map")
    print(f"chair (visualization): {chair_coords_viz if chair_coords_viz else 'Not found'}")

    # door_coords_filtered = client.find_obj_coords("door", top_k=3, score_thres=0.5, filter=True)
    # print(f"door (filtered): {door_coords_filtered if door_coords_filtered else 'Not found'}")
    
    # door_coords_unfiltered = client.find_obj_coords("door", top_k=3, score_thres=0.5, filter=False)
    # print(f"door (unfiltered): {door_coords_unfiltered if door_coords_unfiltered else 'Not found'}")

    # table_coords_regional = client.find_obj_coords(
    #     "table",
    #     current_coords_xy=(10.0, 5.0),
    #     radius=3.0,
    #     top_k=2,
    #     score_thres=0.4
    # )
    # print(f"table (regional, filtered): {table_coords_regional if table_coords_regional else 'Not found'}")
    
    # objects_to_find = ["chair", "window", "nonexistent_object"]
    # multi_object_results = client.find_multi_obj_coords(
    #     objects_to_find,
    #     top_k=2,
    #     score_thres=0.3
    # )
    # print("\nMulti-object (filtered):")
    # for obj_name, coords in multi_object_results.items():
    #     print(f"  {obj_name}: {coords if coords else 'Not found'}")

    # objects_to_find_no_filter = ["chair", "window"]
    # multi_object_results_no_filter = client.find_multi_obj_coords(
    #     objects_to_find_no_filter,
    #     top_k=2,
    #     score_thres=0.3,
    #     filter=False
    # )
    # print("\nMulti-object (unfiltered):")
    # for obj_name, coords in multi_object_results_no_filter.items():
    #     print(f"  {obj_name}: {coords if coords else 'Not found'}")
