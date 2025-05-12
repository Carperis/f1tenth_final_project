#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Point
from visualization_msgs.msg import MarkerArray, Marker
from nav_msgs.msg import Path, Odometry
from geometry_msgs.msg import Pose, PoseStamped
import socket
import json
import os
from datetime import datetime
import getpass

class UDPCommNode(Node):
    def __init__(self):
        super().__init__('udp_comm_node')
        self.declare_parameter('udp_ip', '0.0.0.0')
        self.declare_parameter('udp_port', 5005)
        self.declare_parameter('timer_period', 0.01)

        self.udp_ip = self.get_parameter('udp_ip').get_parameter_value().string_value
        self.udp_port = self.get_parameter('udp_port').get_parameter_value().integer_value
        timer_period = self.get_parameter('timer_period').get_parameter_value().double_value
        self.is_car = getpass.getuser() == 'nvidia'

        self.traj_points = []
        self.publisher_traj = self.create_publisher(MarkerArray, '/trajectory_markers', 10)
        self.publisher_path = self.create_publisher(Path, '/trajectory_path', 10)
        
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind((self.udp_ip, self.udp_port))
        self.sock.settimeout(0.01)

        self.collecting_data = False
        self.current_x = None
        self.current_y = None

        if self.is_car:
            self.pose_sub = self.create_subscription(
                PoseStamped,
                "pf/viz/inferred_pose",
                self.pose_callback,
                10)
            self.get_logger().info("Subscribing to 'pf/viz/inferred_pose' (PoseStamped) for robot pose.")
        else:
            self.pose_sub = self.create_subscription(
                Odometry,
                "/ego_racecar/odom",
                self.pose_callback,
                10)
            self.get_logger().info("Subscribing to '/ego_racecar/odom' (Odometry) for robot pose.")

        self.timer = self.create_timer(timer_period, self.udp_callback)
        
        self.get_logger().info(f"udp_comm_node initialized. Listening on UDP {self.udp_ip}:{self.udp_port}")

    def pose_callback(self, msg):
        if self.is_car: # msg is PoseStamped
            self.current_x = msg.pose.position.x
            self.current_y = msg.pose.position.y
        else: # msg is Odometry
            self.current_x = msg.pose.pose.position.x
            self.current_y = msg.pose.pose.position.y
        # self.get_logger().info(f'Pose updated: x={self.current_x}, y={self.current_y}', throttle_duration_sec=1.0)

    def udp_callback(self):
        self.publish_trajectory()
        try:
            data, addr = self.sock.recvfrom(1024)
            message = data.decode().strip()
            
            if message == "__POSE__":
                if self.current_x is not None and self.current_y is not None:
                    response_message = f"{self.current_x},{self.current_y}"
                    self.sock.sendto(response_message.encode(), addr)
                    self.get_logger().info(f'Reported robot position ({self.current_x}, {self.current_y}) to {addr}', throttle_duration_sec=1.0)
                else:
                    self.sock.sendto(b"None,None", addr)  # Default response if pose is not available
                    self.get_logger().warn('Current pose not available.')
            elif message == "__START__":
                self.get_logger().info('Received START command. Clearing trajectory and starting collection.')
                self.traj_points.clear()
                self.collecting_data = True
                self.last_server_addr = addr # Store server address for reporting
                marker_array_clear = MarkerArray()
                marker_clear = Marker()
                marker_clear.header.frame_id = "map"
                marker_clear.action = Marker.DELETEALL
                marker_array_clear.markers.append(marker_clear)
                self.publisher_traj.publish(marker_array_clear)
            elif message == "__END__":
                self.get_logger().info('Received END command. Stopping collection.')
                self.collecting_data = False
                # Optionally clear self.last_server_addr = None if desired
            elif self.collecting_data:
                try:
                    parts = message.split(',')
                    if len(parts) == 2:
                        x = float(parts[0])
                        y = float(parts[1])
                        
                        point = Point()
                        point.x = x
                        point.y = y
                        point.z = 0.0
                        
                        self.traj_points.append(point)
                        self.get_logger().info(f'Collected trajectory point: ({x}, {y}) from {addr}')
                    else:
                        self.get_logger().warn(f'Invalid point data format: "{message}". Expected "x,y".')
                except ValueError:
                    self.get_logger().warn(f'Could not parse point data: "{message}". Ensure x and y are numbers.')
        
        except socket.timeout:
            pass
        except Exception as e:
            self.get_logger().error(f'Error processing UDP data: {e}')

    def publish_trajectory(self):
        marker_array_msg = MarkerArray()
        path_msg = Path()
        path_msg.header.stamp = self.get_clock().now().to_msg()
        path_msg.header.frame_id = 'map'
        
        for i, p in enumerate(self.traj_points):
            marker = Marker()
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.header.frame_id = 'map'
            marker.ns = "trajectory_points"
            marker.id = i
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD
            marker.pose.position.x = p.x
            marker.pose.position.y = p.y
            marker.pose.position.z = p.z
            marker.pose.orientation.w = 1.0  # No rotation
            marker.scale.x = 0.2  # Sphere diameter
            marker.scale.y = 0.2
            marker.scale.z = 0.2
            marker.color.a = 1.0  # Alpha
            marker.color.r = 0.0  # Red
            marker.color.g = 1.0  # Green
            marker.color.b = 0.0  # Blue
            marker.lifetime = rclpy.duration.Duration(seconds=0).to_msg() # Persist indefinitely, or set a duration
            marker_array_msg.markers.append(marker)
            
            ps = PoseStamped()
            ps.header.stamp = self.get_clock().now().to_msg()
            ps.header.frame_id = 'map'
            ps.pose.position.x = p.x
            ps.pose.position.y = p.y
            ps.pose.position.z = p.z
            ps.pose.orientation.w = 1.0 # Default orientation
            path_msg.poses.append(ps)
        
        if marker_array_msg.markers:
            self.publisher_traj.publish(marker_array_msg)
            
        if path_msg.poses:
            self.publisher_path.publish(path_msg)
        # self.get_logger().info('Trajectory markers published.')

    def destroy_node(self):
        self.publish_trajectory() # Publish one last time if needed
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = UDPCommNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Keyboard interrupt, shutting down.')
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()