import cv2
import numpy as np
import matplotlib.pyplot as plt
import heapq
import yaml  # Add import for yaml

# === A* Node ===
class Node:
    def __init__(self, pt, parent=None, g=0, h=0):
        self.pt = pt          # Tuple (x, y)
        self.parent = parent
        self.g = g            # Cost from start to current node
        self.h = h            # Heuristic cost from current node to goal
        self.f = g + h        # Total cost

    def __lt__(self, other):
        if self.f == other.f:
            return self.h < other.h
        return self.f < other.f

    def __eq__(self, other):
        return self.pt == other.pt

    def __hash__(self):
        return hash(self.pt)

class AStarPlanner:
    def __init__(self, map_file, output_path_file, map_yaml_file, 
                 clearance_radius_m=0.1):
        self.map_file = map_file
        self.output_path_file = output_path_file
        self.map_yaml_file = map_yaml_file
        self.clearance_radius_m = clearance_radius_m

        self.map_img = None
        self.height = 0
        self.width = 0
        self.binary_map = None
        self.planning_map = None
        
        self.start_px = None
        self.goal_px = None
        self.start_world = None
        self.goal_world = None

        self._read_map_config_from_yaml()
        self._load_map_and_create_planning_map()

    def _read_map_config_from_yaml(self):
        with open(self.map_yaml_file, 'r') as f:
            map_config = yaml.safe_load(f)
        self.origin = np.array(map_config['origin'][:2])
        self.resolution = map_config['resolution']
        print(f"Origin set from YAML: {self.origin}")
        print(f"Resolution set from YAML: {self.resolution}")

    def _load_map_and_create_planning_map(self):
        # Load map
        self.map_img = cv2.imread(self.map_file, cv2.IMREAD_GRAYSCALE)
        if self.map_img is None:
            raise FileNotFoundError(f"Map file not found or could not be read: {self.map_file}")
        self.height, self.width = self.map_img.shape
        self.binary_map = (self.map_img > 250).astype(np.uint8)  # 1 for free, 0 for obstacle

        # Create planning map with clearance
        clearance_radius_px = int(self.clearance_radius_m / self.resolution)
        if clearance_radius_px > 0:
            obstacle_mask = 1 - self.binary_map
            kernel_size = 2 * clearance_radius_px + 1
            kernel = np.ones((kernel_size, kernel_size), np.uint8)
            dilated_obstacle_mask = cv2.dilate(obstacle_mask, kernel, iterations=1)
            self.planning_map = 1 - dilated_obstacle_mask
            print(f"Applied clearance of {self.clearance_radius_m}m ({clearance_radius_px}px).")
        else:
            self.planning_map = self.binary_map
            print("Clearance radius is 0 pixels, using original binary map.")

    def _world_to_pixel(self, pt):
        px = int((pt[0] - self.origin[0]) / self.resolution)
        py = int(self.height - (pt[1] - self.origin[1]) / self.resolution)
        return np.array([px, py])

    def _pixel_to_world(self, pt_px):
        x = self.origin[0] + pt_px[0] * self.resolution
        y = self.origin[1] + (self.height - pt_px[1]) * self.resolution
        return np.array([x, y])

    @staticmethod
    def _heuristic(p1, p2):
        return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

    def _is_valid_and_free(self, pt):
        x, y = pt
        if not (0 <= x < self.width and 0 <= y < self.height):
            return False
        return self.planning_map[y, x] == 1

    def _get_neighbors(self, pt):
        neighbors = []
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                neighbor_pt = (pt[0] + dx, pt[1] + dy)
                cost = np.sqrt(dx**2 + dy**2)
                if self._is_valid_and_free(neighbor_pt):
                    neighbors.append((neighbor_pt, cost))
        return neighbors

    def _a_star_search(self):
        open_set = []
        closed_set = set()
        start_node = Node(self.start_px, g=0, h=self._heuristic(self.start_px, self.goal_px))
        heapq.heappush(open_set, start_node)
        g_costs = {self.start_px: 0}

        while open_set:
            current_node = heapq.heappop(open_set)
            if current_node.pt == self.goal_px:
                path = []
                temp = current_node
                while temp:
                    path.append(temp.pt)
                    temp = temp.parent
                return path[::-1]
            if current_node.pt in closed_set:
                continue
            closed_set.add(current_node.pt)

            for neighbor_coords, move_cost in self._get_neighbors(current_node.pt):
                if neighbor_coords in closed_set:
                    continue
                tentative_g_cost = current_node.g + move_cost
                if tentative_g_cost < g_costs.get(neighbor_coords, float('inf')):
                    g_costs[neighbor_coords] = tentative_g_cost
                    h_cost = self._heuristic(neighbor_coords, self.goal_px)
                    neighbor_node = Node(neighbor_coords, parent=current_node, g=tentative_g_cost, h=h_cost)
                    heapq.heappush(open_set, neighbor_node)
        return None

    def _is_line_collision_free(self, p1, p2):
        x1, y1 = p1
        x2, y2 = p2
        dx = abs(x2 - x1)
        dy = abs(y2 - y1)
        sx = 1 if x1 < x2 else -1
        sy = 1 if y1 < y2 else -1
        err = dx - dy

        if not self._is_valid_and_free(p1) or not self._is_valid_and_free(p2):
            return False

        current_x, current_y = x1, y1
        while True:
            if not self._is_valid_and_free((current_x, current_y)):
                return False
            if current_x == x2 and current_y == y2:
                break
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                current_x += sx
            if e2 < dx:
                err += dx
                current_y += sy
        return True

    def _smooth_path_shortcut(self, path):
        if not path or len(path) < 3:
            return path
        smoothed_path = [path[0]]
        current_index = 0
        while current_index < len(path) - 1:
            last_added_point_to_smoothed_path = smoothed_path[-1]
            best_shortcut_index = current_index + 1
            for j in range(len(path) - 1, current_index + 1, -1):
                candidate_point = path[j]
                if self._is_line_collision_free(last_added_point_to_smoothed_path, candidate_point):
                    best_shortcut_index = j
                    break
            smoothed_path.append(path[best_shortcut_index])
            current_index = best_shortcut_index
        return smoothed_path

    def _save_path_to_csv(self, path_to_save_px):
        path_world = np.array([self._pixel_to_world(np.array(p)) for p in path_to_save_px])
        try:
            np.savetxt(self.output_path_file, path_world, delimiter=';', fmt="%.4f", header="x_m;y_m", comments="")
            print(f"Saved path to {self.output_path_file}")
        except Exception as e:
            print(f"Error saving path to CSV: {e}")

    def _visualize_path(self, path_to_visualize_px, original_path_px=None, path_found=True):
        plt.figure()
        plt.imshow(self.map_img, cmap='gray')

        if path_found and path_to_visualize_px:
            if original_path_px and original_path_px != path_to_visualize_px:
                original_path_x_coords = [p[0] for p in original_path_px]
                original_path_y_coords = [p[1] for p in original_path_px]
                plt.plot(original_path_x_coords, original_path_y_coords, 'r--', linewidth=1, label="Original A* Path")

            path_x_coords = [p[0] for p in path_to_visualize_px]
            path_y_coords = [p[1] for p in path_to_visualize_px]
            plt.plot(path_x_coords, path_y_coords, 'b-', linewidth=1.5, label="Smoothed A* Path" if original_path_px else "A* Path")
            title = f"A* Path Planning (Resolution: {self.resolution} m/px)"
        else:
            title = "A* Path Planning - Path Not Found"

        plt.plot(self.start_px[0], self.start_px[1], 'go', markersize=8, label=f"Start_px {self.start_px}")
        plt.plot(self.goal_px[0], self.goal_px[1], 'ro', markersize=8, label=f"Goal_px {self.goal_px}")
        
        plt.legend()
        plt.title(title)
        plt.xlabel("X (pixels)")
        plt.ylabel("Y (pixels)")
        plt.axis('equal')
        plt.grid(True, linestyle='--', alpha=0.5)
        
        plt.show()

    def _find_nearest_free_point(self, pt_px, max_search_radius_px=10):
        """Searches for the nearest free point around pt_px within max_search_radius_px."""
        if self._is_valid_and_free(pt_px):
            return pt_px  # Original point is already free

        for r in range(1, max_search_radius_px + 1):
            for dy in range(-r, r + 1):
                for dx in range(-r, r + 1):
                    # Consider only points on the perimeter of the current search square
                    if abs(dx) != r and abs(dy) != r:
                        continue
                    
                    search_pt = (pt_px[0] + dx, pt_px[1] + dy)
                    if self._is_valid_and_free(search_pt):
                        print(f"Found alternative free goal {search_pt} near original goal {pt_px}")
                        return search_pt
        print(f"Could not find a free point near {pt_px} within radius {max_search_radius_px}")
        return None

    def plan(self, start_world_coords, goal_world_coords, visualize=False, save=False, near=False):
        self.start_world = np.array(start_world_coords)
        self.goal_world = np.array(goal_world_coords)
        self.start_px = tuple(self._world_to_pixel(self.start_world))
        original_goal_px = tuple(self._world_to_pixel(self.goal_world))
        self.goal_px = original_goal_px

        print(f"Starting A* from pixel {self.start_px} (world: {self.start_world}) to pixel {self.goal_px} (world: {self.goal_world})")
        
        if not self._is_valid_and_free(self.start_px):
            print(f"Start point {self.start_px} (world: {self.start_world}) is not valid or is in an obstacle on the planning map.")
            if visualize: self._visualize_path(None, path_found=False)
            return None, None
        
        if not self._is_valid_and_free(self.goal_px):
            print(f"Goal point {self.goal_px} (world: {self.goal_world}) is not valid or is in an obstacle on the planning map.")
            if near:
                print(f"Attempting to find a nearby free goal for {self.goal_px}...")
                new_goal_px = self._find_nearest_free_point(self.goal_px)
                if new_goal_px:
                    self.goal_px = new_goal_px
                    self.goal_world = self._pixel_to_world(np.array(new_goal_px))
                    print(f"Using alternative goal pixel {self.goal_px} (world: {self.goal_world})")
                else:
                    print(f"Could not find a suitable alternative goal near {original_goal_px}.")
                    if visualize: self._visualize_path(None, path_found=False) # Visualize with original goal marked
                    return None, None
            else:
                if visualize: self._visualize_path(None, path_found=False)
                return None, None

        raw_path_px = self._a_star_search()

        if raw_path_px:
            print(f"Path found by A*! Original length: {len(raw_path_px)} points.")
            smoothed_path_px = self._smooth_path_shortcut(raw_path_px)
            print(f"Smoothed path length: {len(smoothed_path_px)} points.")
            
            if save: self._save_path_to_csv(smoothed_path_px)
            if visualize: self._visualize_path(smoothed_path_px, original_path_px=raw_path_px, path_found=True)
            print("A* script finished successfully.")
            return raw_path_px, smoothed_path_px
        else:
            print(f"Failed to find path with A* from {self.start_px} to {self.goal_px}.")
            self._visualize_path(None, path_found=False)
            print("A* script finished with failure.")
            return None, None

# === Main execution ===
if __name__ == "__main__":

    planner = AStarPlanner(
        map_file="/Users/sam/Desktop/Codes/projects_robotics/f1tenth_final_project/maps/map.pgm",
        output_path_file="/Users/sam/Desktop/Codes/projects_robotics/f1tenth_final_project/a_star_path.csv",
        map_yaml_file="/Users/sam/Desktop/Codes/projects_robotics/f1tenth_final_project/maps/map.yaml",
        clearance_radius_m=0.4
    )

    # Define start and goal world coordinates for planning
    start = (9.75 + planner.origin[0], 11.1 + planner.origin[1])
    # goal = (50.1 + planner.origin[0], 26.4 + planner.origin[1])
    # start = (9.75, 11.1)
    # goal = (50.1, 26.4)
    
    from utils import grid2map_coords
    goal = grid2map_coords([(1169, 729)])[0]

    raw_path, smoothed_path = planner.plan(start, goal, visualize=True, save=True, near=True)

    if smoothed_path:
        print(f"Planning complete. Smoothed path has {len(smoothed_path)} points.")
    else:
        print("Planning failed.")
