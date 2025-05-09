import cv2
import numpy as np
import heapq
import matplotlib.pyplot as plt

# Map parameters
resolution = 0.05  # meters/pixel
origin = np.array([0.0, 0.0])  # [x, y]
map_img = cv2.imread("map.pgm", cv2.IMREAD_GRAYSCALE)
height, width = map_img.shape
binary_map = np.uint8(map_img > 250)  # 1 for free, 0 for obstacle

# Load start and goal from CSV (real-world)
start_goal = np.loadtxt("start_goal.csv", delimiter=';', comments="#")
start_world, goal_world = start_goal[0], start_goal[1]

def world_to_map(p_m):
    x_px = int((p_m[0] - origin[0]) / resolution)
    y_px = int(height - (p_m[1] - origin[1]) / resolution)
    return (x_px, y_px)

def map_to_world(p_px):
    x_m = origin[0] + p_px[0] * resolution
    y_m = origin[1] + (height - p_px[1]) * resolution
    return (x_m, y_m)

start_px = world_to_map(start_world)
goal_px = world_to_map(goal_world)

# --- A* Implementation ---
def heuristic(a, b):
    return np.linalg.norm(np.array(a) - np.array(b))

def astar(grid, start, goal):
    neighbors = [(-1,0),(1,0),(0,-1),(0,1), (-1,-1), (1,1), (-1,1), (1,-1)]
    close_set = set()
    came_from = {}
    gscore = {start:0}
    fscore = {start:heuristic(start, goal)}
    oheap = []
    heapq.heappush(oheap, (fscore[start], start))
    
    while oheap:
        _, current = heapq.heappop(oheap)
        if current == goal:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            return path[::-1]
        
        close_set.add(current)
        for dx, dy in neighbors:
            neighbor = (current[0]+dx, current[1]+dy)
            tentative_g_score = gscore[current] + heuristic(current, neighbor)
            if 0 <= neighbor[0] < grid.shape[1] and 0 <= neighbor[1] < grid.shape[0]:
                if grid[neighbor[1]][neighbor[0]] == 0:
                    continue  # obstacle
            else:
                continue  # out of bounds

            if neighbor in close_set and tentative_g_score >= gscore.get(neighbor, float('inf')):
                continue

            if tentative_g_score < gscore.get(neighbor, float('inf')) or neighbor not in [i[1] for i in oheap]:
                came_from[neighbor] = current
                gscore[neighbor] = tentative_g_score
                fscore[neighbor] = tentative_g_score + heuristic(neighbor, goal)
                heapq.heappush(oheap, (fscore[neighbor], neighbor))

    return None

# Run A*
path_px = astar(binary_map, start_px, goal_px)
if path_px is None:
    print("No path found!")
    exit()

# Convert to world coordinates
path_m = [map_to_world(p) for p in path_px]
path_m = np.array(path_m)

# Save path
np.savetxt("astar_path.csv", path_m, delimiter=';', fmt="%.4f", header="x_m;y_m", comments="# ")
print("Path saved to astat_path.csv")

# Plot
plt.figure(figsize=(10,10))
plt.imshow(map_img, cmap='gray')
plt.plot([start_px[0], goal_px[0]], [start_px[1], goal_px[1]], 'go', markersize=8)
path_arr = np.array(path_px)
plt.plot(path_arr[:,0], path_arr[:,1], 'r-', linewidth=2)
plt.title("A* Path Planning")
plt.gca().invert_yaxis()
plt.axis('equal')
plt.grid()
plt.show()
