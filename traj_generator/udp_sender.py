import socket
import csv
import time

JETSON_NANO_IP_ADDRESS = "10.103.119.45"

UDP_IP = JETSON_NANO_IP_ADDRESS
UDP_PORT = 5005

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

with open("/home/shreya/final_project/f1tenth_final_project/traj_generator/mpc_dynamic_trajectory.csv", newline='') as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        line = ';'.join(row)
        sock.sendto(line.encode(), (UDP_IP, UDP_PORT))
        time.sleep(0.04) 

# Optional: send end marker
sock.sendto(b"__END__", (UDP_IP, UDP_PORT))