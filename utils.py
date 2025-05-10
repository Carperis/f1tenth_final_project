import torch.nn.functional as F
from scipy.spatial.transform import Rotation as R
import pandas as pd
import numpy as np
from PIL import Image

import matplotlib.patches as mpatches


def load_pose(pose_file):
    '''
    pose_file: CSV file, each row = [x, y, z, qx, qy, qz, qw]
    '''
    # Read CSV
    df = pd.read_csv(pose_file, header=None)  # <-- no header
    row = df.iloc[0].values  # first pose

    pos = np.array(row[:3], dtype=np.float32).reshape(3, 1)
    quat = row[3:]
    r = R.from_quat(quat)
    rot = r.as_matrix()

    return rot, pos

def load_depth(depth_filepath):
    with open(depth_filepath, "rb") as f:
        depth = np.load(f)
    return depth


def pad_image(img, mean, std, crop_size):
    b, c, h, w = img.shape  # .size()
    assert c == 3
    padh = crop_size - h if h < crop_size else 0
    padw = crop_size - w if w < crop_size else 0
    pad_values = -np.array(mean) / np.array(std)
    img_pad = img.new().resize_(b, c, h + padh, w + padw)
    for i in range(c):
        # note that pytorch pad params is in reversed orders
        img_pad[:, i, :, :] = F.pad(img[:, i, :, :], (0, padw, 0, padh), value=pad_values[i])
    assert img_pad.size(2) >= crop_size and img_pad.size(3) >= crop_size
    return img_pad

def resize_image(img, h, w, **up_kwargs):
    return F.interpolate(img, (h, w), **up_kwargs)


def crop_image(img, h0, h1, w0, w1):
    return img[:, :, h0:h1, w0:w1]

def get_cam_mat_fov(h,w,fov):
    cam_mat = np.eye(3)
    cam_mat[0, 0] = cam_mat[1, 1] = w / (2.0 * np.tan(np.deg2rad(fov / 2)))
    cam_mat[0, 2] = w / 2.0
    cam_mat[1, 2] = h / 2.0
    return cam_mat

def get_cam_mat(h,w):
    cam_mat=np.eye(3)
    cam_mat[0,0]=cam_mat[1,1]=w/2
    cam_mat[0,2]=w/2
    cam_mat[1,2]=h/2
    return cam_mat


def depth2pc(depth, cam_mat=None, fov=90, min_depth=0.1, max_depth=10.0):
    h, w = depth.shape

    if cam_mat is None:
        cam_mat = get_cam_mat_fov(h, w, fov)
    cam_mat_inv = np.linalg.inv(cam_mat)

    y, x = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
    x = x.reshape((1, -1))
    y = y.reshape((1, -1))
    depth = depth.reshape((1, -1))

    p_2d = np.vstack([x, y, np.ones_like(x)])
    pc = cam_mat_inv @ p_2d
    pc = pc * depth

    # Convert to x-forward, y-left, z-up
    pc_converted = np.vstack([
        pc[2, :],       # x = z
        -pc[0, :],      # y = -x
        -pc[1, :]       # z = -y
    ])

    # x is forward → use it as depth
    depth_vals = pc_converted[0, :]
    mask = (depth_vals > min_depth) & (depth_vals < max_depth)

    return pc_converted, mask



def transform_pc(pc,pose):
    pc_homo = np.vstack([pc, np.ones((1,pc.shape[1]))])
    pc_global_homo = pose @ pc_homo
    pc_global = pc_global_homo[:3, :]
    return pc_global
    


def pos2grid_id(gs,cs,xx,yy):
    x = int(gs / 2 + int(xx / cs))
    y = int(gs / 2 - int(yy / cs))
    return [x, y]

def project_point(cam_mat, p_custom):
    """
    Project a 3D point in custom camera frame (x=forward, y=left, z=up)
    to image coordinates using camera intrinsics.

    Args:
        cam_mat (np.ndarray): 3x3 camera intrinsic matrix
        p_custom (np.ndarray): 3D point in (x=forward, y=left, z=up)

    Returns:
        (x_img, y_img, z_cam): pixel coordinates and depth (z in OpenCV frame)
        or None if point is behind the camera
    """
    # Convert custom frame → OpenCV frame
    p_cam = np.array([
        -p_custom[1],  # x = -left → right
        -p_custom[2],  # y = -up   → down
         p_custom[0]   # z = forward (same)
    ]).reshape((3, 1))

    # Project with intrinsics
    new_p = cam_mat @ p_cam
    z = new_p[2, 0]  # depth

    if z <= 0:
        return None  # point behind the camera

    new_p /= z
    x_img = int(new_p[0, 0] + 0.5)
    y_img = int(new_p[1, 0] + 0.5)

    return x_img, y_img, z

def get_new_pallete(num_cls):
    n = num_cls
    pallete = [0] * (n * 3)

    for j in range(0, n):
        lab = j
        pallete[j * 3 + 0] = 0
        pallete[j * 3 + 1] = 0
        pallete[j * 3 + 2] = 0
        i = 0
        while lab > 0:
            pallete[j * 3 + 0] |= ((lab >> 0) & 1) << (7 - i)
            pallete[j * 3 + 1] |= ((lab >> 1) & 1) << (7 - i)
            pallete[j * 3 + 2] |= ((lab >> 2) & 1) << (7 - i)
            i = i + 1
            lab >>= 3
    return pallete


def get_new_mask_pallete(npimg, new_palette, out_label_flag=False, labels=None, ignore_ids_list=[]):
    """Get image color pallete for visualizing masks"""
    # put colormap
    out_img = Image.fromarray(npimg.squeeze().astype("uint8"))
    out_img.putpalette(new_palette)

    if out_label_flag:
        assert labels is not None
        u_index = np.unique(npimg)
        patches = []
        for i, index in enumerate(u_index):
            if index in ignore_ids_list:
                continue
            label = labels[index]
            cur_color = [
                new_palette[index * 3] / 255.0,
                new_palette[index * 3 + 1] / 255.0,
                new_palette[index * 3 + 2] / 255.0,
            ]
            red_patch = mpatches.Patch(color=cur_color, label=label)
            patches.append(red_patch)
    return out_img, patches


def load_map(load_path):
    with open(load_path, "rb") as f:
        map = np.load(f)
    return map


def grid2map_coords(grid_coords, gs=2000, cs=0.1, angle_degrees=63, tx=11, ty=8, tz=0.0):
# def grid2map_coords(grid_coords, gs=2000, cs=0.1, angle_degrees=63, tx=11 - 2.7, ty=8 + 8.14, tz=0.0):
    """
    Converts grid coordinates to map coordinates.
    Args:
        grid_coords: A list of [gx, gy] integer grid coordinates.
        gs (int): Grid size.
        cs (float): Cell size or scaling factor.
        angle_degrees (float): Rotation angle in degrees.
        tx (float): Translation in X.
        ty (float): Translation in Y.
        tz (float, optional): Translation in Z. Defaults to 0.0.
    Returns:
        A list of [mx, my] float map coordinates.
    """
    # Apply scaling and centering
    scaled_points = [[cs * (gx - gs / 2), cs * (gy - gs / 2)] for gx, gy in grid_coords]
    
    # Transformation logic (from transform_points)
    angle_radians = np.deg2rad(angle_degrees)
    cos_theta = np.cos(angle_radians)
    sin_theta = np.sin(angle_radians)

    # Create the 4x4 transformation matrix
    transform_matrix = np.array([
        [cos_theta,  sin_theta, 0,  0],
        [-sin_theta, cos_theta, 0,  0],
        [0,          0,         1,  0],
        [tx,         ty,        tz, 1]
    ])

    # Convert scaled_points to a NumPy array
    points_array = np.array(scaled_points)
    
    # Create homogeneous coordinates: add a z-column (all zeros) and a w-column (all ones)
    num_points = points_array.shape[0]
    homogeneous_points = np.hstack((points_array, np.zeros((num_points, 1)), np.ones((num_points, 1))))
    
    # Apply the transformation to all points at once
    transformed_homogeneous_points = homogeneous_points @ transform_matrix
    
    # Convert back to 2D [x', y'] by taking the first two columns
    map_coords = transformed_homogeneous_points[:, :2].tolist()
    
    return map_coords

def map2grid_coords(map_coords, gs = 2000, cs = 0.1, angle_degrees = 63, tx = 11, ty = 8, tz = 0.0):
# def map2grid_coords(map_coords, gs = 2000, cs = 0.1, angle_degrees = 63, tx=11 - 2.7, ty=8 + 8.14, tz = 0.0):
    """
    Converts map coordinates back to grid coordinates.
    This is the inverse of grid2map_coords.
    Args:
        map_coords (list[list[float]]): A list of [mx, my] float map coordinates.
        gs (int): Grid size.
        cs (float): Cell size or scaling factor.
        angle_degrees (float): Rotation angle in degrees (used in the forward transform).
        tx (float): Translation in X (used in the forward transform).
        ty (float): Translation in Y (used in the forward transform).
        tz (float, optional): Translation in Z (used in the forward transform). Defaults to 0.0.
    Returns:
        list[list[int]]: A list of [gx, gy] integer grid coordinates.
    """
    # 1. Construct the forward transformation matrix T (same as in grid2map_coords)
    angle_radians = np.deg2rad(angle_degrees)
    cos_theta = np.cos(angle_radians)
    sin_theta = np.sin(angle_radians)
    
    transform_matrix_fwd = np.array([
        [cos_theta,  sin_theta, 0,  0],
        [-sin_theta, cos_theta, 0,  0],
        [0,          0,         1,  0],
        [tx,         ty,        tz, 1]
    ])

    # 2. Calculate the inverse transformation matrix T_inv
    try:
        transform_matrix_inv = np.linalg.inv(transform_matrix_fwd)
    except np.linalg.LinAlgError:
        raise ValueError("Transformation matrix is singular and cannot be inverted.")

    # 3. Prepare map_coords for inverse transformation
    map_points_array = np.array(map_coords)
    num_points = map_points_array.shape[0]
    
    # Homogeneous coordinates for map points are [mx, my, tz_forward, 1]
    # because the forward transform results in z_transformed = original_z (0) + tz = tz
    map_homogeneous_points = np.hstack((
        map_points_array,
        np.full((num_points, 1), tz), # Use tz from the forward transform for the z-component
        np.ones((num_points, 1))
    ))

    # 4. Apply the inverse transformation to get scaled_homogeneous_points
    # P_scaled_h = P_map_h @ T_inv
    # This should result in [xs, ys, 0, 1]
    scaled_homogeneous_points = map_homogeneous_points @ transform_matrix_inv

    # Extract scaled 2D points [x_scaled, y_scaled]
    scaled_points_array = scaled_homogeneous_points[:, :2]

    # 5. Reverse scaling and centering
    # x_scaled = cs * (gx - gs / 2)  => gx = (x_scaled / cs) + (gs / 2)
    # y_scaled = cs * (gy - gs / 2)  => gy = (y_scaled / cs) + (gs / 2)
    
    grid_coords_list_float = []
    for x_scaled, y_scaled in scaled_points_array:
        gx = (x_scaled / cs) + (gs / 2.0)
        gy = (y_scaled / cs) + (gs / 2.0)
        grid_coords_list_float.append([gx, gy])

    # 6. Convert to integer grid coordinates by rounding
    grid_coords_list_int = [[int(round(gx)), int(round(gy))] for gx, gy in grid_coords_list_float]
    
    return grid_coords_list_int

def px2map_coords(px_coords, origin=[2.7, -8.14], resolution=0.05, map_height_px=824):
    """
    Converts pixel coordinates (image frame) to map coordinates (world frame).
    Args:
        px_coords (list[list[int]]): List of [px, py] pixel coordinates.
        origin (list[float]): [origin_x, origin_y] of the map in world coordinates.
        resolution (float): Map resolution (meters per pixel).
        map_height_px (int): Height of the map in pixels.
    Returns:
        list[list[float]]: List of [map_x, map_y] map coordinates.
    """
    map_coords_list = []
    for px, py in px_coords:
        map_x = origin[0] + px * resolution
        map_y = origin[1] + (map_height_px - py) * resolution
        map_coords_list.append([map_x, map_y])
    return map_coords_list

def map2px_coords(map_coords, origin, resolution, map_height_px):
    """
    Converts map coordinates (world frame) to pixel coordinates (image frame).
    Args:
        map_coords (list[list[float]]): List of [map_x, map_y] map coordinates.
        origin (list[float]): [origin_x, origin_y] of the map in world coordinates.
        resolution (float): Map resolution (meters per pixel).
        map_height_px (int): Height of the map in pixels.
    Returns:
        list[list[int]]: List of [px, py] pixel coordinates.
    """
    px_coords_list = []
    for map_x, map_y in map_coords:
        px = int(round((map_x - origin[0]) / resolution))
        py = int(round(map_height_px - (map_y - origin[1]) / resolution))
        px_coords_list.append([px, py])
    return px_coords_list

