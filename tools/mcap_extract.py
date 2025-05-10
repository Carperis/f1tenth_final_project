import argparse
import os
import numpy as np
import cv2
import sys

# Import necessary components from mcap
from mcap.reader import make_reader
from mcap_ros2.decoder import DecoderFactory

# Import scipy for transformations
try:
    from scipy.spatial.transform import Rotation as ScipyRotation
except ImportError:
    print("Error: The 'scipy' library is not installed.")
    print("Please install it using: pip install scipy")
    sys.exit(1)


def quaternion_to_matrix(x, y, z, w):
    """Convert quaternion (x, y, z, w) to a 4x4 homogeneous matrix."""
    rotation = ScipyRotation.from_quat([x, y, z, w])
    matrix = np.eye(4)
    matrix[:3, :3] = rotation.as_matrix()
    return matrix

def matrix_to_quaternion_and_translation(matrix):
    """Convert a 4x4 homogeneous matrix to position (x,y,z) and quaternion (x,y,z,w)."""
    rotation_matrix = matrix[:3, :3]
    translation_vector = matrix[:3, 3]

    rotation = ScipyRotation.from_matrix(rotation_matrix)
    quaternion = rotation.as_quat() # Returns [x, y, z, w]

    return translation_vector.tolist(), quaternion.tolist()


def extract_data_from_mcap(mcap_file_path, output_dir, target_fps=10):
    """
    Extracts and syncs color images, depth images, and poses from an MCAP file,
    applies a pose transformation, and saves data in a new structure.

    Args:
        mcap_file_path (str): Path to the input MCAP file.
        output_dir (str): Directory to save the extracted data.
        target_fps (int): The target frames per second for the output data.
    """
    color_topic = '/color_image'
    depth_topic = '/depth_image'
    pose_topic = '/pf/viz/inferred_pose'

    # Ensure main output directory and type subdirectories exist
    output_color_dir = os.path.join(output_dir, 'color')
    output_depth_dir = os.path.join(output_dir, 'depth')
    output_pose_dir = os.path.join(output_dir, 'pose')

    os.makedirs(output_color_dir, exist_ok=True)
    os.makedirs(output_depth_dir, exist_ok=True)
    os.makedirs(output_pose_dir, exist_ok=True)

    print(f"Processing MCAP file: {mcap_file_path}")
    print(f"Saving output to: {output_dir}")
    print(f"Target FPS: {target_fps}")

    # Minimum time interval between saved frames in nanoseconds
    min_interval_ns = int(1e9 / target_fps)

    last_color_msg = None # Stores (decoded_message, log_time_ns)
    last_depth_msg = None # Stores (decoded_message, log_time_ns)
    last_pose_msg = None  # Stores (decoded_message, log_time_ns)

    last_saved_time_ns = 0

    # --- Define the Lidar to Camera Transformation (T_lc) ---
    # Translation: 27cm in pose direction (assume lidar +x), 5cm in z (assume lidar +z)
    translation_lc = np.array([0.27, 0.0, 0.05])

    # Rotation: 20 degrees about the cross product of pose direction (+x) and z axis (assume lidar +z)
    # Cross product of lidar +x (1,0,0) and lidar +z (0,0,1) is (0, -1, 0) (lidar -y axis)
    # Assuming rotation is 20 degrees about the lidar's local -y axis.
    # Note: If the interpretation of the rotation axis or reference frame is different,
    # this part needs adjustment based on clarification.
    rotation_angle_deg = -20
    rotation_axis_lc = np.array([0.0, 1.0, 0.0]) # Lidars local -y axis

    # Create the rotation object and matrix
    rotation_lc = ScipyRotation.from_rotvec(np.deg2rad(rotation_angle_deg) * rotation_axis_lc / np.linalg.norm(rotation_axis_lc))
    rotation_matrix_lc = rotation_lc.as_matrix()

    # Create the 4x4 transformation matrix T_lc
    T_lc = np.eye(4)
    T_lc[:3, :3] = rotation_matrix_lc
    T_lc[:3, 3] = translation_lc
    print(f"Lidar to Camera Transform (T_lc):\n{T_lc}")


    try:
        with open(mcap_file_path, "rb") as f:
            # Use the specified reading approach
            reader = make_reader(f, decoder_factories=[DecoderFactory()])

            # Iterate through messages, getting schema, channel, metadata, and decoded message
            for schema, channel, message, ros_msg in reader.iter_decoded_messages():
                topic = channel.topic
                log_time_ns = message.log_time # Timestamp from the MCAP metadata object

                # Store the latest message for each topic (decoded ROS message and its time)
                if topic == color_topic:
                    last_color_msg = (ros_msg, log_time_ns)
                elif topic == depth_topic:
                    last_depth_msg = (ros_msg, log_time_ns)
                elif topic == pose_topic:
                    last_pose_msg = (ros_msg, log_time_ns)

                # --- Syncing and Saving Logic ---
                # Use the color image as the trigger for saving a frame set
                # Check if enough time has passed since the last save
                if topic == color_topic and (log_time_ns - last_saved_time_ns) >= min_interval_ns:
                    # Check if we have recent data for all three topics
                    # We use the most recently received data for depth and pose
                    if last_color_msg is not None and last_depth_msg is not None and last_pose_msg is not None:
                        # Use the data stored in the last_..._msg variables
                        color_msg_obj, color_time_ns = last_color_msg
                        depth_msg_obj, depth_time_ns = last_depth_msg
                        pose_msg_obj, pose_time_ns = last_pose_msg

                        # Use the color image timestamp for file naming
                        frame_timestamp_ns = color_time_ns

                        # --- Save Color Image (JPG) ---
                        try:
                            # Convert ROS Image message to NumPy array
                            if color_msg_obj.encoding in ['rgb8', 'bgr8', 'mono8']:
                                dtype = np.uint8
                                if color_msg_obj.encoding == 'mono8':
                                    n_channels = 1
                                else:
                                    n_channels = 3
                            elif color_msg_obj.encoding == 'mono16': # Handle mono16 if it somehow appears as color
                                dtype = np.uint16
                                n_channels = 1
                            else:
                                print(f"Warning: Unsupported color encoding '{color_msg_obj.encoding}' at time {color_time_ns}. Skipping color image.")
                                # Skipping this frame set seems safer if color fails
                                continue

                            # Ensure data is bytes
                            img_data = color_msg_obj.data if isinstance(color_msg_obj.data, bytes) else bytes(color_msg_obj.data)

                            # Reshape based on expected channels
                            if n_channels > 1:
                                color_np_array = np.frombuffer(img_data, dtype=dtype).reshape(color_msg_obj.height, color_msg_obj.width, n_channels)
                                # If encoding is rgb8, convert to bgr for OpenCV
                                if color_msg_obj.encoding == 'rgb8':
                                    color_np_array = cv2.cvtColor(color_np_array, cv2.COLOR_RGB2BGR)
                                # If mono8, convert to BGR to save as color JPG (optional)
                                elif color_msg_obj.encoding == 'mono8':
                                     color_np_array = cv2.cvtColor(color_np_array, cv2.COLOR_GRAY2BGR)
                            else: # Mono image
                                color_np_array = np.frombuffer(img_data, dtype=dtype).reshape(color_msg_obj.height, color_msg_obj.width)
                                # Convert mono to BGR for saving as color JPG (optional)
                                if color_msg_obj.encoding == 'mono8' or color_msg_obj.encoding == 'mono16':
                                     color_np_array = cv2.cvtColor(color_np_array, cv2.COLOR_GRAY2BGR)


                            # Save as JPG
                            cv2.imwrite(os.path.join(output_color_dir, f'{frame_timestamp_ns}.jpg'), color_np_array)

                        except Exception as e:
                            print(f"Error processing color image at time {color_time_ns}: {e}")
                            # Skipping this frame set seems safer
                            continue


                        # --- Save Depth Data (NPY) ---
                        try:
                            depth_np_array = True
                             # Convert ROS Image message to NumPy array
                            if depth_msg_obj.encoding == '16UC1' or depth_msg_obj.encoding == 'mono16': # Explicitly handle mono16
                                dtype = np.uint16
                            elif depth_msg_obj.encoding == '32FC1':
                                dtype = np.float32
                            elif depth_msg_obj.encoding == 'mono8': # Sometimes depth is mono8
                                dtype = np.uint8
                            else:
                                print(f"Warning: Unsupported depth encoding '{depth_msg_obj.encoding}' at time {depth_time_ns}. Skipping depth data.")
                                depth_np_array = None # Indicate failure
                                # Let's try to save pose if possible, but warn.
                                pass # Allow pose saving even if depth fails

                            if depth_np_array is not None: # Only process if encoding was supported
                                # Ensure data is bytes
                                depth_data = depth_msg_obj.data if isinstance(depth_msg_obj.data, bytes) else bytes(depth_msg_obj.data)
                                depth_np_array = np.frombuffer(depth_data, dtype=dtype).reshape(depth_msg_obj.height, depth_msg_obj.width)

                                # Save as NPY
                                np.save(os.path.join(output_depth_dir, f'{frame_timestamp_ns}.npy'), depth_np_array)

                        except Exception as e:
                            print(f"Error processing depth image at time {depth_time_ns}: {e}")
                            # Allow pose saving even if depth fails
                            pass


                        # --- Process and Save Transformed Pose Data (CSV) ---
                        try:
                            # Get the original lidar pose (position and quaternion)
                            lidar_pos = pose_msg_obj.pose.position
                            lidar_quat = pose_msg_obj.pose.orientation

                            # Convert lidar pose to a 4x4 matrix (M_l)
                            M_l = quaternion_to_matrix(lidar_quat.x, lidar_quat.y, lidar_quat.z, lidar_quat.w)
                            M_l[:3, 3] = [lidar_pos.x, lidar_pos.y, lidar_pos.z] # Add translation

                            # Apply the transformation: Camera Pose Matrix M_c = M_l * T_lc
                            M_c = M_l @ T_lc

                            # Extract camera pose (position and quaternion) from M_c
                            camera_pos, camera_quat = matrix_to_quaternion_and_translation(M_c)

                            # pose_data format: x,y,z,qx,qy,qz,qw
                            pose_data = f"{camera_pos[0]},{camera_pos[1]},{camera_pos[2]},{camera_quat[0]},{camera_quat[1]},{camera_quat[2]},{camera_quat[3]}\n"

                            with open(os.path.join(output_pose_dir, f'{frame_timestamp_ns}.csv'), 'w') as f_pose:
                                f_pose.write(pose_data)

                        except Exception as e:
                            print(f"Error processing pose data at time {pose_time_ns}: {e}")
                            # Allow saving images even if pose fails
                            pass

                        # Report success and update last saved time
                        print(f"Saved frame set for timestamp: {frame_timestamp_ns} (color: {color_time_ns}, depth: {depth_time_ns}, pose: {pose_time_ns})")
                        last_saved_time_ns = color_time_ns # Update last saved time based on the triggering message

                        # Optional: Clear latest messages after saving to ensure
                        # the next frame uses completely new messages.
                        # last_color_msg = None
                        # last_depth_msg = None
                        # last_pose_msg = None


    except FileNotFoundError:
        print(f"Error: MCAP file not found at {mcap_file_path}")
    except Exception as e:
        print(f"An error occurred while processing the MCAP file: {e}")
        import traceback
        traceback.print_exc()


    print("Extraction complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extracts images and poses from an MCAP file.")
    parser.add_argument("mcap_file", help="Path to the input MCAP file.")
    parser.add_argument("output_directory", help="Directory to save the extracted data.")
    parser.add_argument("--fps", type=int, default=10, help="Target frames per second for the output (default: 10).")

    args = parser.parse_args()

    # Check if required libraries are installed (mcap, mcap-ros2 already checked implicitly by imports)
    try:
        import cv2
    except ImportError:
        print("Error: The 'opencv-python' library is not installed.")
        print("Please install it using: pip install opencv-python")
        exit(1)
    try:
        import numpy as np
    except ImportError:
        print("Error: The 'numpy' library is not installed.")
        print("Please install it using: pip install numpy")
        exit(1)
    # Scipy check is done at the top level import


    extract_data_from_mcap(args.mcap_file, args.output_directory, args.fps)
