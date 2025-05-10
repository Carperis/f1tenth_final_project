import torch.nn.functional as F
from scipy.spatial.transform import Rotation as R
import pandas as pd
import numpy as np
from PIL import Image
from typing import List

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

    for j in 0, n:
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


def grid2map_coords(grid_coords: List[List[int]], gs = 2000, cs = 0.1, angle_degrees = 63, tx = 11, ty = 8, tz = 0.0) -> List[List[float]]:
    """
    Converts grid coordinates (list of lists) to map coordinates (list of lists).
    Args:
        grid_coords (List[List[int]]): A list of [gx, gy] integer grid coordinates.
        gs (int): Grid size.
        cs (float): Cell size or scaling factor.
        angle_degrees (float): Rotation angle in degrees.
        tx (float): Translation in X.
        ty (float): Translation in Y.
        tz (float, optional): Translation in Z. Defaults to 0.0.
    Returns:
        List[List[float]]: A list of [mx, my] float map coordinates.
    """
    if not grid_coords:
        return []
    grid_coords_array = np.array(grid_coords, dtype=float) # Use float for calculations involving cs

    if grid_coords_array.shape[0] == 0:
        return []

    # Apply scaling and centering
    # Assuming grid_coords_array columns are [gx, gy] or [r, c]
    scaled_x = cs * (grid_coords_array[:, 0] - gs / 2.0)
    scaled_y = cs * (grid_coords_array[:, 1] - gs / 2.0)
    scaled_points_array = np.vstack((scaled_x, scaled_y)).T # Shape (N, 2)
    
    angle_radians = np.deg2rad(angle_degrees)
    cos_theta = np.cos(angle_radians)
    sin_theta = np.sin(angle_radians)

    transform_matrix = np.array([
        [cos_theta,  sin_theta, 0,  0],
        [-sin_theta, cos_theta, 0,  0],
        [0,          0,         1,  0],
        [tx,         ty,        tz, 1]
    ])

    num_points = scaled_points_array.shape[0]
    homogeneous_points = np.hstack((scaled_points_array, np.zeros((num_points, 1)), np.ones((num_points, 1))))
    
    transformed_homogeneous_points = homogeneous_points @ transform_matrix
    
    map_coords_array = transformed_homogeneous_points[:, :2]
    
    return map_coords_array.tolist()

def map2grid_coords(map_coords: List[List[float]], gs = 2000, cs = 0.1, angle_degrees = 63, tx = 11, ty = 8, tz = 0.0) -> List[List[int]]:
    """
    Converts map coordinates back to grid coordinates. Vectorized internally.
    Args:
        map_coords (List[List[float]]): A list of [mx, my] float map coordinates.
        gs (int): Grid size.
        cs (float): Cell size or scaling factor.
        angle_degrees (float): Rotation angle in degrees.
        tx (float): Translation in X.
        ty (float): Translation in Y.
        tz (float, optional): Translation in Z. Defaults to 0.0.
    Returns:
        List[List[int]]: A list of [gx, gy] integer grid coordinates.
    """
    if not map_coords:
        return []
        
    map_points_array = np.array(map_coords, dtype=float)
    if map_points_array.shape[0] == 0:
        return []

    angle_radians = np.deg2rad(angle_degrees)
    cos_theta = np.cos(angle_radians)
    sin_theta = np.sin(angle_radians)
    
    transform_matrix_fwd = np.array([
        [cos_theta,  sin_theta, 0,  0],
        [-sin_theta, cos_theta, 0,  0],
        [0,          0,         1,  0],
        [tx,         ty,        tz, 1]
    ])

    try:
        transform_matrix_inv = np.linalg.inv(transform_matrix_fwd)
    except np.linalg.LinAlgError:
        raise ValueError("Transformation matrix is singular and cannot be inverted.")

    num_points = map_points_array.shape[0]
    map_homogeneous_points = np.hstack((
        map_points_array,
        np.full((num_points, 1), tz),
        np.ones((num_points, 1))
    ))
    
    scaled_homogeneous_points = map_homogeneous_points @ transform_matrix_inv
    scaled_points_array = scaled_homogeneous_points[:, :2]

    gx_all = (scaled_points_array[:, 0] / cs) + (gs / 2.0)
    gy_all = (scaled_points_array[:, 1] / cs) + (gs / 2.0)
    
    grid_coords_float_array = np.vstack((gx_all, gy_all)).T
    grid_coords_int_array = np.round(grid_coords_float_array).astype(int)
    
    return grid_coords_int_array.tolist()

def px2map_coords(px_coords: List[List[int]], origin = [2.7, -8.14], resolution = 0.05, map_height_px = 824) -> List[List[float]]:
    """
    Converts pixel coordinates (list of lists, image frame) to map coordinates (list of lists, world frame).
    Args:
        px_coords (List[List[int]]): List of [px, py] pixel coordinates.
        origin (list[float]): [origin_x, origin_y] of the map in world coordinates.
        resolution (float): Map resolution (meters per pixel).
        map_height_px (int): Height of the map in pixels.
    Returns:
        List[List[float]]: List of [map_x, map_y] map coordinates.
    """
    if not px_coords:
        return []
    px_coords_array = np.array(px_coords, dtype=float)

    if px_coords_array.shape[0] == 0:
        return []

    origin_arr = np.array(origin)
    map_x = origin_arr[0] + px_coords_array[:, 0] * resolution
    map_y = origin_arr[1] + (map_height_px - px_coords_array[:, 1]) * resolution
    return np.vstack((map_x, map_y)).T.tolist()

def map2px_coords(map_coords: List[List[float]], origin = [2.7, -8.14], resolution = 0.05, map_height_px = 824) -> List[List[float]]:
    """
    Converts map coordinates (list of lists, world frame) to raw pixel coordinates (list of lists, image frame).
    Args:
        map_coords (List[List[float]]): List of [map_x, map_y] map coordinates.
        origin (list[float]): [origin_x, origin_y] of the map in world coordinates.
        resolution (float): Map resolution (meters per pixel).
        map_height_px (int): Height of the map in pixels.
    Returns:
        List[List[float]]: List of [px_raw, py_raw] raw pixel coordinates (float).
    """
    if not map_coords:
        return []
    map_coords_array = np.array(map_coords, dtype=float)
        
    if map_coords_array.shape[0] == 0:
        return []
        
    origin_arr = np.array(origin)
    px_raw = (map_coords_array[:, 0] - origin_arr[0]) / resolution
    py_raw = map_height_px - (map_coords_array[:, 1] - origin_arr[1]) / resolution
    return np.vstack((px_raw, py_raw)).T.tolist()

