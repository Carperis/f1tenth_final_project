import os
import numpy as np
import cv2
from tqdm import tqdm
import open3d as o3d

# Import necessary functions from utils.py
from utils import load_depth, load_pose, depth2pc, transform_pc

def load_scene_path(data_dir):
    """Loads sorted lists of file paths for RGB, depth, and pose data."""
    rgb_dir = os.path.join(data_dir, "color")
    depth_dir = os.path.join(data_dir, "depth")
    pose_dir = os.path.join(data_dir, "pose")

    # Check if folders exist
    if not (os.path.exists(rgb_dir) and os.path.exists(depth_dir) and os.path.exists(pose_dir)):
        raise FileNotFoundError(f"color, depth, or pose folders are missing in {data_dir}")

    rgb_list = sorted([os.path.join(rgb_dir, f) for f in os.listdir(rgb_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])
    depth_list = sorted([os.path.join(depth_dir, f) for f in os.listdir(depth_dir) if f.endswith('.npy')])
    pose_list = sorted([os.path.join(pose_dir, f) for f in os.listdir(pose_dir) if f.endswith('.csv')])

    # Ensure all lists have the same length
    min_len = min(len(rgb_list), len(depth_list), len(pose_list))
    if not (len(rgb_list) == len(depth_list) == len(pose_list)):
        print(f"Warning: Mismatch in number of files. RGB: {len(rgb_list)}, Depth: {len(depth_list)}, Pose: {len(pose_list)}")
        print(f"Using {min_len} frames.")

    return rgb_list[:min_len], depth_list[:min_len], pose_list[:min_len]

def create_3d_visualization(data_dir, cam_mat, min_depth, max_depth, sample_rate=10):
    """
    Generates and visualizes a combined 3D point cloud from RGB-D data and poses.

    Args:
        data_dir (str): Path to the dataset directory.
        cam_mat (np.ndarray): 3x3 camera intrinsic matrix.
        camera_height (float): Height of the camera from the ground.
        min_depth (float): Minimum valid depth value.
        max_depth (float): Maximum valid depth value.
        sample_rate (int): Rate at which to sample depth pixels (1 = dense).
    """
    rgb_list, depth_list, pose_list = load_scene_path(data_dir)

    if not rgb_list:
        print("Error: No data files found in the specified directory.")
        return

    all_points = []
    all_colors = []
    tf_list = []
    init_tf_inv = None

    pbar = tqdm(range(len(rgb_list)), desc="Generating 3D Map")
    for i, (rgb_path, depth_path, pose_path) in enumerate(zip(rgb_list, depth_list, pose_list)):
        # Load data
        bgr = cv2.imread(rgb_path)
        if bgr is None:
            print(f"Warning: Could not read image {rgb_path}. Skipping frame.")
            pbar.update(1)
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        depth = load_depth(depth_path) * 0.001 # Assuming depth is in mm, convert to meters

        # Assuming load_pose needs pitch_deg, provide a default (e.g., 0)
        # Modify if your load_pose implementation differs
        rot, pos = load_pose(pose_path)

        # Convert robot pose to camera pose (adjust Z based on camera height)
        pose = np.eye(4)
        pose[:3, :3] = rot
        pose[:3, 3] = pos.flatten()

        # Align poses relative to the first frame
        tf_list.append(pose)
        if i == 0:
            init_tf_inv = np.linalg.inv(tf_list[0])
        tf = init_tf_inv @ pose

        # Generate point cloud from depth
        # depth2pc expects depth in original scale (e.g., mm if saved as mm)
        # Let's assume load_depth returns meters or adjust depth2pc call if needed
        pc, mask = depth2pc(depth, cam_mat, min_depth=min_depth, max_depth=max_depth)

        # Get corresponding colors
        h, w = depth.shape
        y_coords, x_coords = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
        y_coords = y_coords.reshape(-1)
        x_coords = x_coords.reshape(-1)

        # Apply mask and sample rate
        valid_indices = np.where(mask)[0]
        if len(valid_indices) == 0:
            pbar.update(1)
            continue # Skip if no valid points

        sampled_indices = valid_indices[::sample_rate] # Sample points

        if len(sampled_indices) == 0:
                pbar.update(1)
                continue # Skip if sampling results in no points

        pc_sampled = pc[:, sampled_indices]
        # Ensure sampled coordinates are within image bounds
        y_coords_sampled = y_coords[sampled_indices]
        x_coords_sampled = x_coords[sampled_indices]
        colors_sampled = rgb[y_coords_sampled, x_coords_sampled] / 255.0 # Normalize colors

        # Transform points to the initial frame
        pc_global = transform_pc(pc_sampled, tf)

        all_points.append(pc_global.T)
        all_colors.append(colors_sampled)
        pbar.update(1)

    pbar.close()

    if not all_points:
        print("No point cloud data generated. Exiting.")
        return

    # Combine all points and colors
    combined_points = np.concatenate(all_points, axis=0)
    combined_colors = np.concatenate(all_colors, axis=0)

    # Create Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(combined_points)
    pcd.colors = o3d.utility.Vector3dVector(combined_colors)

    # Optional: Downsample if the cloud is too large
    # pcd = pcd.voxel_down_sample(voxel_size=0.05) # Adjust voxel size as needed

    # Visualize
    print("Visualizing point cloud... Close the window to exit.")
    o3d.visualization.draw_geometries([pcd])

# --- Script Execution ---
if __name__ == "__main__":
    # --- Configuration ---
    data_path = './data/' # Change to your dataset path if different
    min_depth = 0.3           # Minimum depth value to consider (meters)
    max_depth = 3.0           # Maximum depth value to consider (meters)
    sample_rate = 50          # Use 1 out of every N valid depth points (higher is sparser)

    # Camera intrinsic matrix (replace with your camera's specifics if known)
    cam_mat = np.array([
        [614.7241, 0,       319.6286], # fx, 0, cx
        [0,        614.9275, 241.2029], # 0, fy, cy
        [0,        0,         1     ]  # 0, 0, 1
    ])
    # --- End Configuration ---

    create_3d_visualization(data_path, cam_mat, min_depth, max_depth, sample_rate)
