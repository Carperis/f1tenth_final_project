import requests
import numpy as np
from typing import List, Tuple, Optional
from utils import grid2map_coords, map2grid_coords

# SERVER_URL = "http://localhost:1234/infer"

# def query_location_score_map(location_name):
#     """Queries the server for a score map of the given location name."""
#     try:
#         response = requests.get(SERVER_URL, params={"location_name": location_name})
#         response.raise_for_status()  # Raise an exception for HTTP errors (4xx or 5xx)
#         return response.json()
#     except requests.exceptions.RequestException as e:
#         print(f"Error connecting to server or making request: {e}")
#         return None

def find_top_k_point_coords(data, k=5):
    """Processes the score map to find coordinates above a threshold."""
    if not data or "score_map" not in data:
        print("Invalid data or score_map not found in response.")
        return []

    score_map = np.array(data["score_map"])
    location_name = data.get("location_name", "unknown")
    print(f"Received score map for '{location_name}' with shape: {score_map.shape}")
    
    top_k_flat_indices = np.argsort(score_map.flatten())[-k:]
    rows, cols = np.unravel_index(top_k_flat_indices, score_map.shape)
    top_k_coords = list(zip(rows.tolist(), cols.tolist()))
    print(f"Top {k} coordinates for '{location_name}': {top_k_coords}")
    return top_k_coords

def find_high_score_coords(data, score_threshold=10):
    """Finds coordinates with scores above a certain threshold."""
    if not data or "score_map" not in data:
        print("Invalid data or score_map not found in response.")
        return []

    score_map = np.array(data["score_map"])
    high_scores = np.argwhere(score_map > score_threshold)
    high_scores_list = high_scores.tolist()
    print(f"Coordinates with scores above {score_threshold}: {high_scores_list}")
    return high_scores_list

def find_object_coordinates(
    obj_name: str,
    current_coords_xy: Optional[Tuple[int, int]] = None,
    radius: Optional[float] = None,
    top_k: Optional[int] = None,
    score_thres: float = 0.0
) -> List[Tuple[int, int]]:
    """
    Queries a server for an object's score map and finds relevant coordinates.

    Coordinates are returned as (x, y), corresponding to (column, row) in the score map.

    Args:
        obj_name (str): The name of the object to query.
        current_coords_xy (Optional[Tuple[int, int]]): Current coordinates in (x, y) format.
                                                       If provided, a regional search is performed.
        radius (Optional[float]): Radius for the regional search around current_coords_rc.
                                   Only used if current_coords_rc is also provided.
        top_k (Optional[int]): Number of top results to return, sorted by score.
                               If None or invalid, all matching coordinates (after other filters) are returned.
        score_thres (float): Minimum score for a coordinate to be considered (inclusive).

    Returns:
        List[Tuple[int, int]]: A list of (x, y) coordinates, where x is column and y is row.
                               Returns an empty list if errors occur or no coordinates match.
    """
    
    # SERVER_URL = "http://10.103.107.67:1234/infer"
    SERVER_URL = "http://127.0.0.1:1234/infer"
    
    def query_location_score_map(location_name):
        """Queries the server for a score map of the given location name."""
        try:
            response = requests.get(SERVER_URL, params={"location_name": location_name})
            response.raise_for_status()  # Raise an exception for HTTP errors (4xx or 5xx)
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Error connecting to server or making request: {e}")
            return None
    
    map_data = query_location_score_map(obj_name)

    if map_data is None:
        # query_location_score_map already prints an error
        return []

    if "error" in map_data:
        print(f"Server returned an error for '{obj_name}': {map_data['error']}")
        return []

    if "score_map" not in map_data or map_data["score_map"] is None:
        print(f"No 'score_map' data found or it is null in the server response for '{obj_name}'.")
        return []
    
    score_map_data = map_data["score_map"]
    try:
        score_map = np.array(score_map_data)
    except Exception as e:
        print(f"Could not convert score_map data to NumPy array for '{obj_name}': {e}")
        return []

    if score_map.ndim != 2:
        print(f"Received score_map for '{obj_name}' is not 2-dimensional. Actual shape: {score_map.shape}")
        return []

    # 1. Get all coordinates (row, col) and their scores that meet the threshold
    candidate_indices_rc = np.argwhere(score_map >= score_thres)
    
    coords_rc_with_scores = []
    for r_idx, c_idx in candidate_indices_rc:
        coords_rc_with_scores.append(((int(r_idx), int(c_idx)), score_map[r_idx, c_idx]))

    # 2. Filter by radius if current_coords_rc and radius are provided and valid
    if current_coords_xy is not None:
        current_coords_cr = map2grid_coords([current_coords_xy])[0]
        current_coords_rc = (int(current_coords_cr[1]), int(current_coords_cr[0]))
        
        grid_radius = radius / 0.1
        
        if grid_radius is not None and isinstance(grid_radius, (int, float)) and grid_radius >= 0:
            curr_r, curr_c = current_coords_rc
            region_filtered_coords = []
            for (r, c), score in coords_rc_with_scores:
                dist = np.sqrt((r - curr_r)**2 + (c - curr_c)**2)
                if dist <= grid_radius:
                    region_filtered_coords.append(((r, c), score))
            coords_rc_with_scores = region_filtered_coords
        elif grid_radius is None:
             print(f"Warning: Regional search for '{obj_name}' requested (current_coords_rc provided) but no radius was given. Skipping regional filter.")
        else: # Invalid radius
            print(f"Warning: Invalid radius ({grid_radius}) provided for regional search of '{obj_name}'. Regional filter skipped.")
    
    # 3. Sort by score in descending order
    coords_rc_with_scores.sort(key=lambda item: item[1], reverse=True)

    # 4. Select top_k if specified and valid
    final_coords_rc_with_scores = coords_rc_with_scores
    if top_k is not None:
        if isinstance(top_k, int) and top_k > 0:
            final_coords_rc_with_scores = coords_rc_with_scores[:top_k]
        else:
            print(f"Warning: Invalid top_k value ({top_k}) for '{obj_name}'. Returning all currently filtered results.")
            # final_coords_rc_with_scores remains as all sorted & filtered coords
        
    result_coords_rc = [[r, c] for (r, c), score in final_coords_rc_with_scores]
    result_coords_xy = grid2map_coords(result_coords_rc)
    
    return result_coords_xy


if __name__ == "__main__":
    # --- Test with a specific location ---
    # Make sure the map_inference_server.py is running before executing this client.
    # location_to_query = "door"  # Example: query for "door"
    # print(f"Querying server for location: '{location_to_query}'...")
    
    # map_data = query_location_score_map(location_to_query)

    # if map_data:
    #     if "error" in map_data:
    #         print(f"Server returned an error: {map_data['error']}")
    #     else:
    #         print("Successfully received data from server.")
    #         # Find top k coordinates
    #         top_k_coords = find_top_k_point_coords(map_data, k=5)
    #         # Find high score coordinates
    #         high_score_coords = find_high_score_coords(map_data, score_threshold=14)

    # print("\nClient finished.")
    
    location_to_query = "door"
    print(f"Querying server for location: '{location_to_query}'...")
    # result = find_object_coordinates(location_to_query, top_k=5, score_thres=10, current_coords_xy=(15.3, 3.7), radius=5)
    result = find_object_coordinates(location_to_query, top_k=5, score_thres=10)
    if result:
        print(f"{result}")
    else:
        print("No doors found with specified criteria.")
    
    
