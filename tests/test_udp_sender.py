import socket
import csv
import time
import numpy as np # Added for point generation
from utils import grid2map_coords, map2grid_coords

JETSON_NANO_IP_ADDRESS = "10.103.86.254"

UDP_IP = JETSON_NANO_IP_ADDRESS
UDP_PORT = 5005

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# float_points = []
# with open('a_star_path.csv', 'r') as file:
    # csv_reader = csv.reader(file, delimiter=';')
    # next(csv_reader)  # Skip header row
    # for row in csv_reader:
    #     if row: # ensure row is not empty
    #         try:
    #             x = float(row[0])
    #             y = float(row[1])
    #             float_points.append((x, y))
    #         except ValueError as e:
    #             print(f"Skipping row due to ValueError: {row} - {e}")
    #         except IndexError as e:
    #             print(f"Skipping row due to IndexError: {row} - {e}")

raw_points = [(1000, 1000), (1169, 729), (1005, 934), (986, 925), (1049, 841), (1170, 729)]
# raw_points = [(1055, 881), (1053, 889), (1078, 870), (1049, 886), (1053, 881)]
float_points = grid2map_coords(raw_points)

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

sock.sendto(b"__POSE__", (UDP_IP, UDP_PORT))
data, addr = sock.recvfrom(1024)  # Buffer size is 1024 bytes
print(f"Received feedback: {data.decode()} from {addr}")


# # Add code to receive feedback
# print("Waiting for feedback...")
# try:
#     sock.settimeout(10.0)  # Set a timeout for receiving data (e.g., 10 seconds)
#     data, addr = sock.recvfrom(1024)  # Buffer size is 1024 bytes
#     print(f"Received feedback: {data.decode()} from {addr}")
# except socket.timeout:
#     print("No feedback received within the timeout period.")
# finally:
#     sock.close()