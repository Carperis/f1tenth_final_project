import numpy as np
import matplotlib.pyplot as plt
import cv2
import math

# Parameters
resolution = 0.05  # meters per pixel
origin = np.array([0.0, 0.0])
L = 0.33
MAX_STEER = 0.4189
MAX_ACCEL = 3.0
MAX_SPEED = 5.0
DT = 0.1
N = 8

# Load map
map_img = cv2.imread('map.pgm', cv2.IMREAD_GRAYSCALE)
height, width = map_img.shape

# Load path
waypoints = np.loadtxt('path.csv', delimiter=';', comments='#')
headings = np.arctan2(np.gradient(waypoints[:, 1]), np.gradient(waypoints[:, 0]))
speeds = np.full(len(waypoints), 2.0)

# Initialize state
x, y, yaw, v = waypoints[0, 0], waypoints[0, 1], headings[0], 0.0
trajectory = [[x, y, v, yaw]]
ref_index = 0

# Simulate trajectory
for t in range(1000):
    if ref_index >= len(waypoints) - N - 1:
        break

    target = waypoints[ref_index + 1]
    dx, dy = target[0] - x, target[1] - y
    target_yaw = math.atan2(dy, dx)
    yaw_err = math.atan2(math.sin(target_yaw - yaw), math.cos(target_yaw - yaw))
    steer = np.clip(2.0 * yaw_err, -MAX_STEER, MAX_STEER)
    acc = np.clip(1.0 * (speeds[ref_index] - v), -MAX_ACCEL, MAX_ACCEL)

    # Kinematic update
    x += v * math.cos(yaw) * DT
    y += v * math.sin(yaw) * DT
    yaw += v / L * math.tan(steer) * DT
    v += acc * DT
    v = np.clip(v, 0.0, MAX_SPEED)
    trajectory.append([x, y, v, yaw])

    if np.linalg.norm([x - target[0], y - target[1]]) < 0.5:
        ref_index += 1

trajectory = np.array(trajectory)

# Convert world to pixel for plotting
def world_to_pixel(pt):
    px = int((pt[0] - origin[0]) / resolution)
    py = int(height - (pt[1] - origin[1]) / resolution)
    return px, py

trajectory_px = np.array([world_to_pixel(pt[:2]) for pt in trajectory])
waypoints_px = np.array([world_to_pixel(pt) for pt in waypoints])

# Plot on map
fig, ax = plt.subplots(figsize=(10, 10))
ax.imshow(map_img, cmap='gray')
ax.plot(waypoints_px[:, 0], waypoints_px[:, 1], 'r--', label='Reference Path')
ax.plot(trajectory_px[:, 0], trajectory_px[:, 1], 'b-', label='MPC Trajectory')
ax.scatter(*world_to_pixel(waypoints[0]), c='green', label='Start')
ax.scatter(*world_to_pixel(waypoints[-1]), c='purple', label='Goal')
ax.invert_yaxis()
ax.set_title("MPC Trajectory on Map")
ax.legend()
plt.grid()
plt.axis('equal')
plt.savefig("mpc_on_map_final.png", dpi=300)

"mpc_on_map_final.png"
