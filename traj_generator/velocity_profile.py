import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Load trajectory file
trajectory_data = np.loadtxt("mpc_trajectory.csv", delimiter=';', skiprows=1)

# Estimate velocity using position differences
positions = trajectory_data[:, :2]
velocities = [0.0]
for i in range(1, len(positions)):
    dx = positions[i][0] - positions[i-1][0]
    dy = positions[i][1] - positions[i-1][1]
    v = np.sqrt(dx**2 + dy**2) / 0.1  # assuming constant dt=0.1s
    velocities.append(v)

# Plot velocity profile
plt.figure(figsize=(10, 5))
plt.plot(velocities, label='Velocity (m/s)', color='blue')
plt.title("MPC Velocity Profile Over Time")
plt.xlabel("Timestep")
plt.ylabel("Velocity (m/s)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("mpc_velocity_profile.png", dpi=300)

# Return path to file
"mpc_velocity_profile.png"
