import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cp
import math
import cv2

# === Constants ===
DT = 0.1           # Time step
L = 0.33           # Wheelbase [m]
N = 10              # Horizon steps
MAX_SPEED = 5.0    # [m/s]
MAX_ACCEL = 3.0    # [m/s^2]
MAX_STEER = 0.4189 # [rad]
RESOLUTION = 0.05  # [m/pixel]
ORIGIN = np.array([0.0, 0.0])

map_img = cv2.imread("map.pgm", cv2.IMREAD_GRAYSCALE)
height, width = map_img.shape
path = np.loadtxt("path.csv", delimiter=';', comments='#')
assert path.shape[1] == 2, "path.csv must contain two columns: x_m;y_m"

def wrap_to_pi(angle):
    return math.atan2(math.sin(angle), math.cos(angle))

def heading_between(p1, p2):
    return math.atan2(p2[1] - p1[1], p2[0] - p1[0])

def find_nearest_index(x, y, path):
    dists = np.linalg.norm(path - np.array([x, y]), axis=1)
    return np.argmin(dists)

def get_linearized_matrices(v, theta, delta):
    A = np.eye(4)
    A[0, 3] = math.cos(theta) * DT
    A[1, 3] = math.sin(theta) * DT
    A[2, 3] = math.tan(delta) / L * DT

    B = np.zeros((4, 2))
    B[3, 0] = DT
    B[2, 1] = v * DT / (L * (math.cos(delta) ** 2))

    f = np.array([
        v * math.cos(theta),
        v * math.sin(theta),
        v * math.tan(delta) / L,
        0.0
    ])
    C = (f - A @ np.zeros(4) - B @ np.zeros(2)) * DT
    return A, B, C

# === MPC Setup ===
Q = np.diag([10, 10, 1, 1])       # state deviation cost
Qf = np.diag([20, 20, 2, 2])      # final state cost
R = np.diag([0.01, 0.1])          # input effort
Rd = np.diag([0.1, 0.2])          # input rate change

# === Simulation Setup ===
x0 = np.array([path[0, 0], path[0, 1], heading_between(path[0], path[1]), 0.0])
x = x0.copy()
trajectory = [x.copy()]
velocities = [x[3]]
last_u = np.zeros(2)
ref_index = 0

for step in range(2000):
    if ref_index >= len(path) - N - 1:
        break

    # === Setup reference trajectory ===
    ref_traj = []
    for i in range(N + 1):
        idx = min(ref_index + i, len(path) - 1)
        px, py = path[idx]
        theta = heading_between(path[idx - 1], path[idx]) if idx > 0 else x[2]
        v = 2.0  # desired speed
        ref_traj.append([px, py, theta, v])
    ref_traj = np.array(ref_traj)

    # === Variables ===
    z = cp.Variable((4, N + 1))
    u = cp.Variable((2, N))

    cost = 0
    constraints = [z[:, 0] == x]

    for t in range(N):
        A, B, C = get_linearized_matrices(x[3], x[2], last_u[1])
        cost += cp.quad_form(z[:, t] - ref_traj[t], Q)
        cost += cp.quad_form(u[:, t], R)
        if t > 0:
            cost += cp.quad_form(u[:, t] - u[:, t - 1], Rd)

        constraints += [
            z[:, t + 1] == A @ z[:, t] + B @ u[:, t] + C,
            cp.abs(u[0, t]) <= MAX_ACCEL,
            cp.abs(u[1, t]) <= MAX_STEER,
            cp.norm(z[3, t + 1], 'inf') <= MAX_SPEED
        ]

    cost += cp.quad_form(z[:, N] - ref_traj[N], Qf)

    prob = cp.Problem(cp.Minimize(cost), constraints)
    prob.solve(solver=cp.OSQP)

    if prob.status != cp.OPTIMAL:
        print("MPC failed to solve!")
        break

    u_star = u.value[:, 0]
    last_u = u_star.copy()

    acc, delta = u_star
    x[0] += x[3] * math.cos(x[2]) * DT
    x[1] += x[3] * math.sin(x[2]) * DT
    x[2] += x[3] * math.tan(delta) / L * DT
    x[2] = wrap_to_pi(x[2])
    x[3] += acc * DT
    x[3] = np.clip(x[3], 0.0, MAX_SPEED)

    trajectory.append(x.copy())
    velocities.append(x[3])
    if np.linalg.norm(path[ref_index] - x[:2]) < 0.5:
        ref_index += 1
trajectory = np.array(trajectory)

def world_to_pixel(pt):
    px = int((pt[0] - ORIGIN[0]) / RESOLUTION)
    py = int(height - (pt[1] - ORIGIN[1]) / RESOLUTION)
    return px, py

path_px = np.array([world_to_pixel(p) for p in path])
traj_px = np.array([world_to_pixel(p[:2]) for p in trajectory])

plt.figure(figsize=(10, 10))
plt.imshow(map_img, cmap='gray')
plt.plot(path_px[:, 0], path_px[:, 1], 'r--', label='Reference Path')
plt.plot(traj_px[:, 0], traj_px[:, 1], 'b-', label='MPC Trajectory')
plt.scatter(*world_to_pixel(path[0]), c='green', label='Start')
plt.scatter(*world_to_pixel(path[-1]), c='purple', label='Goal')
plt.legend()
plt.axis('equal')
plt.title("MPC Path on Map")
plt.grid(True)
plt.gca().invert_yaxis()
plt.savefig("mpc_dynamic_final.png", dpi=300)

# === Plot Velocity Profile ===
plt.figure()
plt.plot(velocities)
plt.xlabel("Time Step")
plt.ylabel("Velocity [m/s]")
plt.title("Velocity Profile")
plt.grid(True)
plt.savefig("velocity_profile_dynamic.png", dpi=300)

# === Save final trajectory ===
np.savetxt("mpc_dynamic_trajectory.csv", trajectory[:, :2], delimiter=';', fmt="%.4f", header="x_m;y_m", comments='')