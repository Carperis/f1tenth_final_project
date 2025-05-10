# Create visualize_poses.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.spatial.transform import Rotation as R
import argparse

def load_all_poses(pose_file, pitch_deg):
    """
    Load all poses from a CSV file and apply inverse pitch correction.

    Args:
        pose_file (str): Path to the CSV file. Each row = [x, y, z, qx, qy, qz, qw].
        pitch_deg (float): Pitch angle in degrees to correct for.

    Returns:
        list: A list of tuples, where each tuple contains (rotation_matrix, position_vector).
    """
    try:
        df = pd.read_csv(pose_file, header=None)
    except pd.errors.EmptyDataError:
        print(f"Error: Pose file '{pose_file}' is empty or invalid.")
        return []
    except FileNotFoundError:
        print(f"Error: Pose file '{pose_file}' not found.")
        return []

    poses = []
    pitch_rad = -np.deg2rad(pitch_deg)
    R_pitch_inv = np.array([
        [np.cos(pitch_rad), 0, np.sin(pitch_rad)],
        [0, 1, 0],
        [-np.sin(pitch_rad), 0, np.cos(pitch_rad)]
    ])

    for index, row in df.iterrows():
        values = row.values
        if len(values) < 7:
            print(f"Warning: Skipping row {index} due to insufficient data.")
            continue
        try:
            pos = np.array(values[:3], dtype=np.float32).reshape(3, 1)
            quat = values[3:7] # Ensure only 4 values are taken for quaternion
            r = R.from_quat(quat)
            rot = r.as_matrix()

            # Apply inverse pitch correction
            rot_corrected = R_pitch_inv @ rot
            poses.append((rot_corrected, pos))
        except Exception as e:
            print(f"Warning: Skipping row {index} due to error: {e}")
            continue

    return poses

def main(pose_file, pitch_deg, output_video, axis_length=1.0, interval=50): # Default axis_length changed to 1.0
    """
    Generates a video visualizing camera poses.

    Args:
        pose_file (str): Path to the pose CSV file.
        pitch_deg (float): Pitch angle in degrees for correction.
        output_video (str): Path to save the output MP4 video.
        axis_length (float): Length of the coordinate axes representing the camera.
        interval (int): Delay between frames in milliseconds.
    """
    poses = load_all_poses(pose_file, pitch_deg)
    if not poses:
        print("No valid poses loaded. Exiting.")
        return

    positions = np.array([p[1].flatten() for p in poses])

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Determine plot limits
    max_range = np.array([positions[:, 0].max()-positions[:, 0].min(),
                          positions[:, 1].max()-positions[:, 1].min(),
                          positions[:, 2].max()-positions[:, 2].min()]).max() / 2.0

    mid_x = (positions[:, 0].max()+positions[:, 0].min()) * 0.5
    mid_y = (positions[:, 1].max()+positions[:, 1].min()) * 0.5
    mid_z = (positions[:, 2].max()+positions[:, 2].min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    ax.set_xlabel('X (World)')
    ax.set_ylabel('Y (World)')
    ax.set_zlabel('Z (World)')
    ax.set_title('Camera Pose Trajectory')

    # Plot trajectory path
    ax.plot(positions[:, 0], positions[:, 1], positions[:, 2], 'grey', label='Trajectory')

    # Initialize lines for camera axes (X=Red, Y=Green, Z=Blue)
    # Using standard robotics convention: X-forward, Y-left, Z-up
    # Need to map this to matplotlib's convention if different
    origin = np.zeros((3, 1))
    x_axis = np.array([[axis_length], [0], [0]])
    y_axis = np.array([[0], [axis_length], [0]]) # Matplotlib Y is typically horizontal left/right
    z_axis = np.array([[0], [0], [axis_length]]) # Matplotlib Z is typically vertical

    # Initial pose
    rot_init, pos_init = poses[0]
    x_end_init = pos_init + rot_init @ x_axis
    y_end_init = pos_init + rot_init @ y_axis
    z_end_init = pos_init + rot_init @ z_axis

    line_x, = ax.plot([pos_init[0,0], x_end_init[0,0]], [pos_init[1,0], x_end_init[1,0]], [pos_init[2,0], x_end_init[2,0]], 'r-', linewidth=2)
    line_y, = ax.plot([pos_init[0,0], y_end_init[0,0]], [pos_init[1,0], y_end_init[1,0]], [pos_init[2,0], y_end_init[2,0]], 'g-', linewidth=2)
    line_z, = ax.plot([pos_init[0,0], z_end_init[0,0]], [pos_init[1,0], z_end_init[1,0]], [pos_init[2,0], z_end_init[2,0]], 'b-', linewidth=2)
    title = ax.text2D(0.05, 0.95, "", transform=ax.transAxes)

    # Update function for animation
    def update(frame):
        rot, pos = poses[frame]

        # Calculate axis endpoints in world frame
        x_end = pos + rot @ x_axis
        y_end = pos + rot @ y_axis
        z_end = pos + rot @ z_axis

        # Update X axis line
        line_x.set_data([pos[0,0], x_end[0,0]], [pos[1,0], x_end[1,0]])
        line_x.set_3d_properties([pos[2,0], x_end[2,0]])

        # Update Y axis line
        line_y.set_data([pos[0,0], y_end[0,0]], [pos[1,0], y_end[1,0]])
        line_y.set_3d_properties([pos[2,0], y_end[2,0]])

        # Update Z axis line
        line_z.set_data([pos[0,0], z_end[0,0]], [pos[1,0], z_end[1,0]])
        line_z.set_3d_properties([pos[2,0], z_end[2,0]])

        title.set_text(f'Frame {frame}/{len(poses)-1}')

        return line_x, line_y, line_z, title

    # Create animation
    ani = FuncAnimation(fig, update, frames=len(poses),
                        interval=interval, blit=True, repeat=False)

    # Show the plot window
    plt.show()

    # Save animation
    try:
        print(f"Saving animation to {output_video}...")
        ani.save(output_video, writer='ffmpeg', fps=1000/interval)
        print("Animation saved successfully.")
    except Exception as e:
        print(f"Error saving animation: {e}")
        print("Ensure ffmpeg is installed and accessible in your system's PATH.")
        # plt.show() # No longer needed here

    plt.close(fig) # Close the plot figure


if __name__ == "__main__":
    # python viz_all_poses.py path/to/pose_file.csv --pitch 0.0 --output pose_visualization.mp4 --length 2.0 --interval 50
    
    parser = argparse.ArgumentParser(description="Visualize camera poses from a CSV file as a video.")
    parser.add_argument("pose_file", help="Path to the CSV file containing poses (x, y, z, qx, qy, qz, qw per row).")
    parser.add_argument("--pitch", type=float, default=0.0, help="Pitch angle correction in degrees (default: 0.0).")
    parser.add_argument("--output", default="pose_visualization.mp4", help="Output video file name (default: pose_visualization.mp4).")
    parser.add_argument("--length", type=float, default=2.0, help="Length of the visualized camera axes (default: 1.0).") # Default axis_length changed to 1.0
    parser.add_argument("--interval", type=int, default=50, help="Delay between frames in milliseconds (default: 50ms -> 20 FPS).")

    args = parser.parse_args()

    main(args.pose_file, args.pitch, args.output, args.length, args.interval)