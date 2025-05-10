import socket
import csv
import time
from utils import grid2map_coords, map2grid_coords
import os # Added for os.path.exists

DEFAULT_IP_ADDRESS = "127.0.0.1" # Default IP, can be overridden
DEFAULT_UDP_PORT = 5005

class UDPComm:
    def __init__(self, udp_ip=DEFAULT_IP_ADDRESS, udp_port=DEFAULT_UDP_PORT, timeout=10.0, period=0.1):
        self.udp_ip = udp_ip
        self.udp_port = udp_port
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.timeout = timeout # Timeout for receiving data
        self.period = period # Period for sending data

    def _send_points_list(self, points_list):
        """Helper function to send a list of (x,y) float points."""
        try:
            self.sock.sendto(b"__START__", (self.udp_ip, self.udp_port))
            print(f"Sending {len(points_list)} points to UDP server at {self.udp_ip}:{self.udp_port}...")
            for x, y in points_list:
                line = f"{x:.2f},{y:.2f}"
                self.sock.sendto(line.encode(), (self.udp_ip, self.udp_port))
                # print(f"Sent: {line}") # Optional: for debugging
                time.sleep(self.period)
            self.sock.sendto(b"__END__", (self.udp_ip, self.udp_port))
            print("Finished sending points.")
            return True
        except Exception as e:
            print(f"Error sending points: {e}")
            return False

    def send_trajectory_from_csv(self, csv_path, is_grid=False):
        """
        Sends trajectory from a CSV file.
        CSV format: x;y (or x,y - delimiter will be auto-detected or specified)
        is_grid: If True, converts coordinates from grid to map.
        Returns: True if successful, False otherwise.
        """
        float_points = []
        try:
            with open(csv_path, 'r') as file:
                # Attempt to sniff the delimiter
                try:
                    dialect = csv.Sniffer().sniff(file.read(1024))
                    file.seek(0)
                    csv_reader = csv.reader(file, dialect)
                except csv.Error:
                    # Default to semicolon if sniffing fails
                    file.seek(0)
                    csv_reader = csv.reader(file, delimiter=';')
                
                header = next(csv_reader, None)  # Skip header row if present
                if header and (header[0].lower().strip() == 'x' and header[1].lower().strip() == 'y'):
                    print(f"Skipped header row: {header}")
                else: # If not a typical header, rewind and process as data
                    file.seek(0) # Go back to the beginning of the file
                    csv_reader = csv.reader(file, delimiter=';' if 'dialect' not in locals() else dialect.delimiter)


                for row in csv_reader:
                    if row and len(row) >= 2: # ensure row is not empty and has at least two values
                        try:
                            x = float(row[0].strip())
                            y = float(row[1].strip())
                            float_points.append((x, y))
                        except ValueError as e:
                            print(f"Skipping row due to ValueError: {row} - {e}")
                            continue # Skip this row and continue with the next
                    elif row:
                        print(f"Skipping malformed row (not enough columns): {row}")
            
            if not float_points:
                print("No valid points found in CSV file.")
                return False

            if is_grid:
                # Assuming the points from CSV are grid coordinates
                # utils.grid2map_coords expects a list of tuples [(x1,y1), (x2,y2), ...]
                map_points = grid2map_coords(float_points)
            else:
                map_points = float_points
            
            return self._send_points_list(map_points)

        except FileNotFoundError:
            print(f"Error: CSV file not found at {csv_path}")
            return False
        except Exception as e:
            print(f"An error occurred while processing CSV {csv_path}: {e}")
            return False

    def send_trajectory_from_list(self, coords_list, is_grid=True):
        """
        Sends trajectory from a list of coordinates.
        coords_list: A list of [x,y].
        is_grid: If True, converts coordinates from grid to map.
        Returns: True if successful, False otherwise.
        """
            
        try:
            if is_grid:
                # utils.grid2map_coords expects a list of tuples [(x1,y1), (x2,y2), ...]
                map_points = grid2map_coords(coords_list)
            else:
                map_points = coords_list
            
            return self._send_points_list(map_points)
        except Exception as e:
            print(f"An error occurred while sending trajectory from list: {e}")
            return False

    def get_car_position(self):
        """
        Requests and retrieves the car's current position [x,y].
        Returns: [float, float] list if successful, None otherwise.
        """
        try:
            self.sock.sendto(b"__POSE__", (self.udp_ip, self.udp_port))
            # print("Requested car position...")
            
            self.sock.settimeout(self.timeout) 
            data, addr = self.sock.recvfrom(1024)  # Buffer size is 1024 bytes
            decoded_data = data.decode().strip()
            # print(f"Received raw pose data: '{decoded_data}' from {addr}")

            if decoded_data.lower() == "none,none" or not decoded_data:
                # print("Received None,None or empty data for position.")
                return None
            
            parts = decoded_data.split(',')
            if len(parts) == 2:
                x = float(parts[0])
                y = float(parts[1])
                return [x, y]
            else:
                print(f"Received malformed position data: {decoded_data}")
                return None
        except socket.timeout:
            print(f"Timeout: No position data received from {self.udp_ip} within {self.timeout} seconds.")
            return None
        except ValueError as e:
            print(f"Error converting position data to float: {decoded_data} - {e}")
            return None
        except Exception as e:
            print(f"Error getting car position: {e}")
            return None
        finally:
            self.sock.settimeout(None) # Reset timeout

    def close_socket(self):
        """Closes the UDP socket."""
        print("Closing UDP socket.")
        self.sock.close()

if __name__ == '__main__':
    # Example Usage (assuming a UDP server is listening)
    # Make sure utils.py and a sample CSV are in the same directory or accessible

    test_csv_path = "/Users/sam/Desktop/Codes/projects_robotics/f1tenth_final_project/a_star_path.csv" 

    # Initialize UDPComm
    # Replace with actual IP if testing with a real Jetson Nano
    comm = UDPComm(udp_ip="10.103.74.128") 

    print("\n--- Testing send_trajectory_from_csv (is_grid=False) ---")
    if os.path.exists(test_csv_path): # Check if the test file exists
        success_csv = comm.send_trajectory_from_csv(test_csv_path, is_grid=False)
        print(f"CSV send success: {success_csv}")
    else:
        print(f"Test CSV file '{test_csv_path}' not found. Skipping CSV send test.")

    # print("\n--- Testing send_trajectory_from_list (is_grid=True) ---")
    # raw_points_list = [(1169, 729), (1005, 934), (986, 925), (1049, 841), (1170, 729)]
    # success_list = comm.send_trajectory_from_list(raw_points_list, is_grid=True)
    # print(f"List send success: {success_list}")
    
    # print("\n--- Testing send_trajectory_from_list (is_grid=False) ---")
    # map_points_list = grid2map_coords(raw_points_list)
    # success_list_map = comm.send_trajectory_from_list(map_points_list, is_grid=False)
    # print(f"List send (map coords) success: {success_list_map}")

    # print("\n--- Testing get_car_position ---")
    # # This part requires a server to be running and sending back a response
    # # To test this, you'd need a simple UDP server listening on 127.0.0.1:5005
    # # that responds to "__POSE__" with "x,y"
    # position = comm.get_car_position()
    # if position:
    #     print(f"Received car position: {position}")
    # else:
    #     print("Failed to get car position or position was None.")

    # Clean up
    comm.close_socket()
    print("\nExample usage finished.")
