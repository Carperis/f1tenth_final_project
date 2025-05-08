import socket
import csv
import time
import numpy as np # Added for point generation

def transform_points(points_2d, angle_degrees, tx, ty, tz=0.0):
    """
    Transforms a list of 2D points using a 4x4 matrix for rotation (around Z) and translation.
    Args:
        points_2d (list of lists/tuples): The initial [x, y] points.
        angle_degrees (float): Rotation angle in degrees.
        tx (float): Translation in X.
        ty (float): Translation in Y.
        tz (float, optional): Translation in Z. Defaults to 0.0.
    Returns:
        list of lists: The transformed [x, y] points.
    """
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

    # Convert points_2d to a NumPy array
    points_array = np.array(points_2d)
    
    # Create homogeneous coordinates: add a z-column (all zeros) and a w-column (all ones)
    # Input points_array is N x 2. We want N x 4.
    num_points = points_array.shape[0]
    homogeneous_points = np.hstack((points_array, np.zeros((num_points, 1)), np.ones((num_points, 1))))
    
    # Apply the transformation to all points at once
    # P_transformed_homogeneous (N x 4) = P_homogeneous (N x 4) @ transform_matrix (4 x 4)
    transformed_homogeneous_points = homogeneous_points @ transform_matrix
    
    # Convert back to 2D [x', y'] by taking the first two columns
    transformed_points_list = transformed_homogeneous_points[:, :2].tolist()
    
    return transformed_points_list

JETSON_NANO_IP_ADDRESS = "10.103.81.99"

UDP_IP = JETSON_NANO_IP_ADDRESS
UDP_PORT = 5005

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# points = []
# with open('points.csv', 'r') as file:
#     csv_reader = csv.reader(file)
#     for row in csv_reader:
#         points.append(row[:2])
gs = 2000
cs = 0.1
# raw_points = [(1000, 1000), (1169, 729), (1005, 934), (986, 925), (1049, 841), (1170, 729)]
raw_points = [(1055, 881), (1053, 889), (1078, 870), (1049, 886), (1053, 881)]
float_points = [[cs*(gx-gs/2), cs*(gy-gs/2)] for gx, gy in raw_points]
float_points = transform_points(float_points, 63, 11, 8, 0)

points = [[f"{x:.2f}", f"{y:.2f}"] for x, y in float_points]

sock.sendto(b"__START__", (UDP_IP, UDP_PORT))
print("Sending points to UDP server...")
for p in points:
    line = ','.join(p)
    sock.sendto(line.encode(), (UDP_IP, UDP_PORT))
    print(f"Sent: {line}")
    time.sleep(1)  # Adjust the delay as needed
sock.sendto(b"__END__", (UDP_IP, UDP_PORT))
print("Finished sending points.")