import cv2
import numpy as np
import matplotlib.pyplot as plt
import random
from collections import deque
from skimage.draw import line as sk_line

# === Parameters ===
resolution = 0.05
origin = np.array([0.0, 0.0])
map_file = "map.pgm"
start_goal_file = "start_goal.csv"
max_iters = 10000
step_size = 10  # in pixels
goal_threshold = 15  # in pixels

# === Load map and binary version ===
map_img = cv2.imread(map_file, cv2.IMREAD_GRAYSCALE)
height, width = map_img.shape
binary_map = (map_img > 250).astype(np.uint8)

# === Convert world to pixel ===
def world_to_pixel(pt):
    px = int((pt[0] - origin[0]) / resolution)
    py = int(height - (pt[1] - origin[1]) / resolution)
    return np.array([px, py])

# === Convert pixel to world ===
def pixel_to_world(pt_px):
    x = origin[0] + pt_px[0] * resolution
    y = origin[1] + (height - pt_px[1]) * resolution
    return np.array([x, y])

# === Load start and goal ===
start_world, goal_world = np.loadtxt(start_goal_file, delimiter=';')
start_px = world_to_pixel(start_world)
goal_px = world_to_pixel(goal_world)

# === RRT core ===
class Node:
    def __init__(self, pt, parent=None):
        self.pt = pt
        self.parent = parent

def is_free(pt):
    x, y = pt
    return 0 <= x < width and 0 <= y < height and binary_map[y, x] == 1

def line_is_free(p1, p2):
    rr, cc = sk_line(p1[1], p1[0], p2[1], p2[0])  # row = y, col = x
    for y, x in zip(rr, cc):
        if not (0 <= x < width and 0 <= y < height):
            return False
        if binary_map[y, x] == 0:
            return False
    return True

def distance(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))

def nearest_node(nodes, pt):
    return min(nodes, key=lambda node: distance(node.pt, pt))

def steer(p1, p2, step):
    vec = np.array(p2) - np.array(p1)
    norm = np.linalg.norm(vec)
    if norm < step:
        return tuple(p2)
    direction = vec / norm
    new_pt = np.array(p1) + step * direction
    return tuple(np.round(new_pt).astype(int))

# === RRT algorithm ===
tree = [Node(tuple(start_px))]
for _ in range(max_iters):
    if random.random() < 0.1:
        sample = tuple(goal_px)
    else:
        sample = (random.randint(0, width - 1), random.randint(0, height - 1))

    nearest = nearest_node(tree, sample)
    new_pt = steer(nearest.pt, sample, step_size)

    if is_free(new_pt) and line_is_free(nearest.pt, new_pt):
        new_node = Node(new_pt, parent=nearest)
        tree.append(new_node)

        if distance(new_pt, goal_px) < goal_threshold:
            print("Path found!")
            goal_node = Node(tuple(goal_px), parent=new_node)
            break
else:
    raise RuntimeError("Failed to find path in RRT")

# === Backtrack path ===
path = []
node = goal_node
while node is not None:
    path.append(node.pt)
    node = node.parent
path = path[::-1]

# === Save path to CSV in world coordinates ===
path_world = np.array([pixel_to_world(np.array(p)) for p in path])
np.savetxt("path.csv", path_world, delimiter=';', fmt="%.4f", header="x_m;y_m", comments="")
print("Saved RRT path to path.csv")

# === Visualize ===
plt.figure(figsize=(10, 10))
plt.imshow(map_img, cmap='gray')
plt.plot([p[0] for p in path], [p[1] for p in path], 'r-', linewidth=2, label="RRT Path")
plt.plot(start_px[0], start_px[1], 'go', label="Start")
plt.plot(goal_px[0], goal_px[1], 'bo', label="Goal")
plt.legend()
plt.title("RRT Path Planning")
plt.gca().invert_yaxis()
plt.axis('equal')
plt.grid()
plt.savefig("rrt_path.png", dpi=300, bbox_inches='tight')
plt.show()
 