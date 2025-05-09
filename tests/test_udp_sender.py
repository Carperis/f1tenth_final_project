import socket
import csv
import time
import numpy as np # Added for point generation
from utils import grid2map_coords, map2grid_coords

JETSON_NANO_IP_ADDRESS = "10.103.81.99"

UDP_IP = JETSON_NANO_IP_ADDRESS
UDP_PORT = 5005

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# points = []
# with open('points.csv', 'r') as file:
#     csv_reader = csv.reader(file)
#     for row in csv_reader:
#         points.append(row[:2])

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