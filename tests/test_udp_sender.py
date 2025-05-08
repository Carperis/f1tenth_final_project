import socket
import csv
import time
import numpy as np # Added for point generation

JETSON_NANO_IP_ADDRESS = "10.103.76.66"

UDP_IP = JETSON_NANO_IP_ADDRESS
UDP_PORT = 5005

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

points = []
with open('points.csv', 'r') as file:
    csv_reader = csv.reader(file)
    for row in csv_reader:
        points.append(row[:2])

sock.sendto(b"__START__", (UDP_IP, UDP_PORT))
print("Sending points to UDP server...")
for p in points:
    line = ','.join(p)
    sock.sendto(line.encode(), (UDP_IP, UDP_PORT))
    print(f"Sent: {line}")
    time.sleep(0.01)  # Adjust the delay as needed
sock.sendto(b"__END__", (UDP_IP, UDP_PORT))
print("Finished sending points.")