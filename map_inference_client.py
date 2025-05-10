import requests
import numpy as np
from typing import List, Tuple, Optional, Dict
from utils import grid2map_coords, map2grid_coords, map2px_coords, px2map_coords

class MapInferenceClient:
    def __init__(self, server_url: str = "http://127.0.0.1:1234/infer"):
        self.server_url = server_url

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
        score_thres: float = 0.0
    ) -> List[List[float]]:
        map_data = self._query_location_score_map(obj_name)

        if not map_data or "score_map" not in map_data or map_data["score_map"] is None:
            return []
        
        score_map = np.array(map_data["score_map"])

        if score_map.ndim != 2:
            return []

        candidate_indices_rc = np.argwhere(score_map >= score_thres)
        
        coords_rc_with_scores = [
            ((int(r_idx), int(c_idx)), score_map[r_idx, c_idx])
            for r_idx, c_idx in candidate_indices_rc
        ]

        if current_coords_xy and radius is not None and radius >= 0:
            current_coords_cr_list = map2grid_coords([current_coords_xy])
            if current_coords_cr_list:
                current_coords_cr = current_coords_cr_list[0]
                current_coords_rc = (int(current_coords_cr[1]), int(current_coords_cr[0]))
                
                map_resolution = 0.1 # Assuming 0.1 m/pixel
                grid_radius = radius / map_resolution
            
                curr_r, curr_c = current_coords_rc
                coords_rc_with_scores = [
                    item for item in coords_rc_with_scores
                    if np.sqrt((item[0][0] - curr_r)**2 + (item[0][1] - curr_c)**2) <= grid_radius
                ]
        
        coords_rc_with_scores.sort(key=lambda item: item[1], reverse=True)

        if top_k is not None and top_k > 0:
            coords_rc_with_scores = coords_rc_with_scores[:top_k]
            
        result_coords_rc = [[r, c] for (r, c), _ in coords_rc_with_scores]
        
        if not result_coords_rc:
            return []
            
        result_coords_xy_tuples = grid2map_coords(result_coords_rc)
        return [[float(x), float(y)] for x, y in result_coords_xy_tuples]

    def find_multi_obj_coords(
        self,
        obj_names: List[str],
        current_coords_xy: Optional[Tuple[float, float]] = None,
        radius: Optional[float] = None,
        top_k: Optional[int] = None,
        score_thres: float = 0.0
    ) -> Dict[str, List[List[float]]]:
        results: Dict[str, List[List[float]]] = {}
        for obj_name in obj_names:
            results[obj_name] = self.find_obj_coords(
                obj_name,
                current_coords_xy,
                radius,
                top_k,
                score_thres
            )
        return results

if __name__ == '__main__':
    client = MapInferenceClient()

    print("--- Testing single object: 'door' ---")
    door_coords = client.find_obj_coords("door", top_k=3, score_thres=0.5)
    print(f"Found door coordinates: {door_coords}" if door_coords else "No door coordinates found.")

    print("\n--- Testing single object: 'table' with regional search ---")
    table_coords_regional = client.find_obj_coords(
        "table",
        current_coords_xy=(10.0, 5.0),
        radius=3.0,
        top_k=2,
        score_thres=0.4
    )
    print(f"Found table coordinates (regional): {table_coords_regional}" if table_coords_regional else "No table coordinates found (regional).")
    
    print("\n--- Testing multiple objects: ['chair', 'window', 'nonexistent_object'] ---")
    objects_to_find = ["chair", "window", "nonexistent_object"]
    multi_object_results = client.find_multi_obj_coords(
        objects_to_find,
        top_k=2,
        score_thres=0.3
    )
    for obj_name, coords in multi_object_results.items():
        print(f"Coordinates for '{obj_name}': {coords}" if coords else f"No coordinates found for '{obj_name}'.")
    print("\nClient tests finished.")
