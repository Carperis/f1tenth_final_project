import socket
import csv
import time
from utils import grid2map_coords, map2grid_coords
import os

DEFAULT_IP_ADDRESS = "127.0.0.1"
DEFAULT_UDP_PORT = 5005

class UDPComm:
    def __init__(self, udp_ip=DEFAULT_IP_ADDRESS, udp_port=DEFAULT_UDP_PORT, timeout=10.0, period=0.1):
        self.udp_ip = udp_ip
        self.udp_port = udp_port
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.timeout = timeout
        self.period = period

    def _send_points_list(self, points_list):
        """Helper function to send a list of (x,y) float points."""
        self.sock.sendto(b"__START__", (self.udp_ip, self.udp_port))
        for x, y in points_list:
            line = f"{x:.2f},{y:.2f}"
            self.sock.sendto(line.encode(), (self.udp_ip, self.udp_port))
            time.sleep(self.period)
        self.sock.sendto(b"__END__", (self.udp_ip, self.udp_port))
        return True

    def send_trajectory_from_csv(self, csv_path, is_grid=False):
        """
        Sends trajectory from a CSV file.
        CSV format: x;y or x,y.
        is_grid: If True, converts coordinates from grid to map.
        """
        float_points = []
        if not os.path.exists(csv_path):
            return False

        with open(csv_path, 'r') as file:
            try:
                dialect = csv.Sniffer().sniff(file.read(1024))
                file.seek(0)
                csv_reader = csv.reader(file, dialect)
            except csv.Error:
                file.seek(0)
                csv_reader = csv.reader(file, delimiter=';')
            
            header = next(csv_reader, None)
            if not (header and header[0].lower().strip() == 'x' and header[1].lower().strip() == 'y'):
                file.seek(0)
                try:
                    dialect = csv.Sniffer().sniff(file.read(1024))
                    file.seek(0)
                    csv_reader = csv.reader(file, dialect)
                except csv.Error:
                    file.seek(0)
                    csv_reader = csv.reader(file, delimiter=';')

            for row in csv_reader:
                if row and len(row) >= 2:
                    x = float(row[0].strip())
                    y = float(row[1].strip())
                    float_points.append((x, y))
        
        if not float_points:
            return False

        map_points = grid2map_coords(float_points) if is_grid else float_points
        return self._send_points_list(map_points)

    def send_trajectory_from_list(self, coords_list, is_grid=False):
        """
        Sends trajectory from a list of coordinates.
        coords_list: A list of [x,y].
        is_grid: If True, converts coordinates from grid to map.
        """
        map_points = grid2map_coords(coords_list) if is_grid else coords_list
        return self._send_points_list(map_points)

    def get_car_position(self):
        """
        Requests and retrieves the car's current position [x,y].
        Returns: [float, float] list if successful, None otherwise.
        """
        self.sock.sendto(b"__POSE__", (self.udp_ip, self.udp_port))
        self.sock.settimeout(self.timeout)
        data, addr = self.sock.recvfrom(1024)
        self.sock.settimeout(None)
        
        decoded_data = data.decode().strip()

        if decoded_data.lower() == "None,None" or not decoded_data:
            return None
        
        parts = decoded_data.split(',')
        if len(parts) == 2:
            x = float(parts[0])
            y = float(parts[1])
            return [x, y]
        return None

    def close_socket(self):
        """Closes the UDP socket."""
        self.sock.close()

if __name__ == '__main__':
    comm = UDPComm(udp_ip="127.0.0.1")
    
    test_csv_path = "a_star_path_0.csv"
    if os.path.exists(test_csv_path):
        success_csv = comm.send_trajectory_from_csv(test_csv_path, is_grid=False)

    raw_points_list = [(1169, 729), (1005, 934)]
    success_list = comm.send_trajectory_from_list(raw_points_list, is_grid=True)

    comm.close_socket()
