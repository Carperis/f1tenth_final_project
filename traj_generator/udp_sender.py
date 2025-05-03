import socket
import csv
import time

JETSON_NANO_IP_ADDRESS = "10.103.119.45"

UDP_IP = JETSON_NANO_IP_ADDRESS
UDP_PORT = 5005

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

with open("mpc_dynamic_trajectory.csv", newline='') as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        line = ','.join(row)
        sock.sendto(line.encode(), (UDP_IP, UDP_PORT))
        time.sleep(0.1)  # send at 10 Hz; adjust as needed

# Optional: send end marker
sock.sendto(b"__END__", (UDP_IP, UDP_PORT))