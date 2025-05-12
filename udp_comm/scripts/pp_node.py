#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import numpy as np
from nav_msgs.msg import Path
from ackermann_msgs.msg import AckermannDriveStamped
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point, PoseStamped
from nav_msgs.msg import Odometry
from scipy.spatial.transform import Rotation as R
import csv
import getpass
import time
import udp_comm.reeds_shepp as rs

IS_CAR = getpass.getuser() == 'nvidia'

# Configuration for Reeds Shepp
ROBOT_SPEED = 0.7  # m/s (constant speed for path execution)
MAX_STEER = 0.25 # Steering angle at which to do turns
ROBOT_WHEELBASE = 0.5 # m (Assumed wheelbase for Ackermann steering calculation)

class PurePursuit(Node):
    """
    Advanced Pure Pursuit controller with multiple lookahead distances
    for steering and speed control
    """

    def __init__(self):
        super().__init__("pp_node")

        self.declare_parameters(
            namespace="",
            parameters=[
                ("lookahead_distance", 1.2),  # Steering lookahead (blue)
                ("speed_lookahead_distance", 1.0),  # Speed reduction lookahead (red)
                ("turn_lookahead_distance", 1.0),  # Speed reduction lookahead (red)
                ("base_velocity", 2.0),  # Base target speed
                ("brake_gain", 2.0),  # Seed reduction gain
                ("curvature_thresh", 1.0),  # Curvature threshold
                ("curvature_limit_gain", 0.6),  # Curvature limit gain
                ("wheel_base", 0.33),  # Vehicle wheelbase
            ],
        )

        # ==============
        # ROS Infrastructure
        # ==============
        if IS_CAR:
            self.sub_pose = self.create_subscription(PoseStamped, "pf/viz/inferred_pose", self.pose_callback, 10)
        else:
            self.sub_pose = self.create_subscription(Odometry, "/ego_racecar/odom", self.pose_callback, 10)
        self.sub_path = self.create_subscription(Path, "/trajectory_path", self.path_callback, 10)

        self.pub_drive = self.create_publisher(AckermannDriveStamped, "drive", 10)
        self.pub_ld_marker = self.create_publisher(MarkerArray, "lookahead_markers", 10)
        self.pub_path_marker = self.create_publisher(MarkerArray, "path_marker", 10)

        self.path_marker_id = 0
        self.steer_marker_id = 1
        self.speed_marker_id = 2
        self.max_steering_degree = 30

        self.waypoints = np.array([])
        self.num_waypoints = 0

        self.path_marker_array = MarkerArray()
        self.current_yaw = 0
        self.turn_hysteresis = 0

        self.get_logger().info(f"pp_node initialized.")

    def find_lookahead_index(self, start_idx, lookahead_dist):
        """
        Find the waypoint that is lookahead_dist away along the path.
        This considers the cumulative distance along the path and handles circular paths.
        """
        cumulative_distance = 0.0
        idx = start_idx

        while cumulative_distance < lookahead_dist:
            # Compute the distance to the next waypoint
            next_idx = (idx + 1) % self.num_waypoints
            segment_distance = np.linalg.norm(self.waypoints[next_idx] - self.waypoints[idx])

            # Add the segment distance to the cumulative distance
            cumulative_distance += segment_distance

            # Move to the next waypoint
            idx = next_idx

            # If we've looped back to the start, break to avoid infinite loops
            if idx == start_idx:
                break

        return idx

    def publish_drive_command(self, linear_velocity, steering_angle):
        """Publishes an AckermannDriveStamped message."""
        msg = AckermannDriveStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.drive.speed = linear_velocity
        msg.drive.steering_angle = steering_angle
        self.pub_drive.publish(msg)

    def execute_turn(self, start_state, end_state):
        """
        Computes a Reeds-Shepp path to the target angle and executes the commands.
        The robot will turn in place by a sepcified number of degrees.
        """
        self.get_logger().info(f"Calculating path from {start_state} to {end_state}")

        path_elements = rs.get_optimal_path(start_state, end_state)

        if path_elements is None:
            self.get_logger().warn("No Reed-Shepp path found for the requested orientation change.")
            return # Exit function if no path found

        self.get_logger().info(f"Found path with {len(path_elements)} elements.")

        # Execute path elements
        for i, element in enumerate(path_elements):
            self.get_logger().info(f"Executing element {i+1}/{len(path_elements)}: Steering={element.steering}, Gear={element.gear}, Param={element.param:.2f}")

            linear_velocity = ROBOT_SPEED if element.gear == rs.Gear.FORWARD else -ROBOT_SPEED
            steering_angle_rad = 0.0 # Default to straight

            if element.steering == rs.Steering.STRAIGHT:
                 steering_angle_rad = 0.0 # Straight segment
                 distance = element.param
            elif element.steering == rs.Steering.LEFT or element.steering == rs.Steering.RIGHT:
                # For turns, element.param is the angle turned in radians
                # The radius of the turn segment is the turning_radius used in get_optimal_path
                # Steering angle for Ackermann is atan(L / R)
                steering_angle_rad = MAX_STEER
                steering_radius = ROBOT_WHEELBASE / np.tan(steering_angle_rad)
                distance = element.param * steering_radius
                if element.steering == rs.Steering.RIGHT:
                    steering_angle_rad *= -1 # Negative for right turns

            # Calculate duration for this segment
            # Use element.param as the length of the segment
            duration = distance / ROBOT_SPEED if ROBOT_SPEED > 1e-9 else 0.0

            if duration > 0:
                # Publish the command
                self.publish_drive_command(linear_velocity, steering_angle_rad)
                # Wait for the duration of the command
                time.sleep(duration) # Using time.sleep

            # Stop the robot briefly after each segment (optional, can help with precision)
            # self.publish_drive_command(0.0, 0.0)
            # time.sleep(0.1) # Small delay

        # After executing all segments, the robot should be at the target orientation
        self.publish_drive_command(0.0, 0.0)

    def path_callback(self, msg):
        original_waypoints = np.array([[float(pose.pose.position.x), float(pose.pose.position.y)] for pose in msg.poses])
        num_original_waypoints = len(original_waypoints)

        last_n = len(self.waypoints)
        last_start = self.waypoints[0][0] if len(self.waypoints) > 0 else 0

        if num_original_waypoints < 2:
            self.waypoints = original_waypoints
            self.num_waypoints = num_original_waypoints
        else:
            target_total_points = 100
            x_orig = original_waypoints[:, 0]
            y_orig = original_waypoints[:, 1]

            # Calculate cumulative distances along the original path
            cumulative_dist_orig = np.zeros(num_original_waypoints)
            for k in range(1, num_original_waypoints):
                cumulative_dist_orig[k] = cumulative_dist_orig[k-1] + np.linalg.norm(original_waypoints[k] - original_waypoints[k-1])

            if cumulative_dist_orig[-1] == 0: # Path has zero length (all points are the same)
                self.waypoints = np.array([original_waypoints[0]] * target_total_points)
            else:
                # Create a new set of distances for the 1000 points
                dist_new = np.linspace(0, cumulative_dist_orig[-1], target_total_points)

                # Interpolate x and y coordinates
                x_interp = np.interp(dist_new, cumulative_dist_orig, x_orig)
                y_interp = np.interp(dist_new, cumulative_dist_orig, y_orig)

                self.waypoints = np.vstack((x_interp, y_interp)).T
            self.num_waypoints = len(self.waypoints)

        self.path_marker_array = MarkerArray()
        self.path_marker_array.markers.append(self.create_path_marker())

        # self.get_logger().info(f"Received path with {len(self.waypoints)} waypoints")

    def pose_callback(self, pose_msg):
        if self.num_waypoints < 1:
            self.get_logger().warn("No waypoints available. Waiting for path...")
            return
        # print(f"# of waypoints: {self.num_waypoints}")

        # Retrieve parameters
        self.ld_steer = self.get_parameter("lookahead_distance").value
        self.ld_speed = self.get_parameter("speed_lookahead_distance").value
        self.ld_turn = self.get_parameter("turn_lookahead_distance").value
        self.base_vel = self.get_parameter("base_velocity").value
        self.brake_gain = self.get_parameter("brake_gain").value
        self.wheel_base = self.get_parameter("wheel_base").value
        self.curvature_thresh = self.get_parameter("curvature_thresh").value
        self.cur_limit_gain = self.get_parameter("curvature_limit_gain").value

        # ==============
        # Current State
        # ==============
        if IS_CAR:
            current_pos = np.array([pose_msg.pose.position.x, pose_msg.pose.position.y])
            orientation = [pose_msg.pose.orientation.x, pose_msg.pose.orientation.y, pose_msg.pose.orientation.z, pose_msg.pose.orientation.w]
        else:
            current_pos = np.array([pose_msg.pose.pose.position.x, pose_msg.pose.pose.position.y])
            orientation = [pose_msg.pose.pose.orientation.x, pose_msg.pose.pose.orientation.y, pose_msg.pose.pose.orientation.z, pose_msg.pose.pose.orientation.w]
        self.current_orientation = orientation[2]

        x,y,z,w = orientation
        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        self.current_yaw = np.rad2deg(np.arctan2(siny_cosp, cosy_cosp))

        self.enable_turn = self.turn_hysteresis > 10
        if self.enable_turn:
            print('pos at turn end', (current_pos[0], current_pos[1], self.current_yaw))
            self.turn_hysteresis = 0

        # ==============
        # Waypoint Selection
        # ==============
        closest_idx = np.argmin(np.linalg.norm(self.waypoints - current_pos, axis=1))

        # Find three different lookahead points
        steer_idx = self.find_lookahead_index(closest_idx, self.ld_steer)
        speed_idx = self.find_lookahead_index(closest_idx, self.ld_speed)
        turn_idx = self.find_lookahead_index(closest_idx, self.ld_speed)

        # ==============
        # Coordinate Transforms
        # ==============
        rot_matrix = R.from_quat(orientation).as_matrix().T  # Inverse rotation
        current_pos_3d = np.append(current_pos, 0)  # Add z-coordinate

        def transform_point(point):
            """Transform a world point to vehicle coordinates"""
            vec_to_point = point - current_pos_3d[:2]
            return rot_matrix[:2, :2] @ vec_to_point

        # Transform all lookahead points
        steer_point = transform_point(self.waypoints[steer_idx])
        speed_point = transform_point(self.waypoints[speed_idx])
        turn_point = transform_point(self.waypoints[turn_idx])

        # ==============
        # Steering Calculation
        # ==============
        lateral_displacement = steer_point[1]
        curvature = (2 * lateral_displacement) / (self.ld_steer**2)
        steering_angle = np.arctan(self.wheel_base * curvature) # angle = arctan(L / R) where curvature = 1 / R

        # ==============
        # Velocity Control
        # ==============
        # Speed control based on speed lookahead point
        speed_heading = np.arctan2(speed_point[1], speed_point[0])
        brake_amount = self.brake_gain * abs(speed_heading)

        # Combine velocity components
        target_velocity_raw = self.base_vel - brake_amount

        # Curvature-based speed limiting
        if abs(curvature) > self.curvature_thresh:
            target_velocity = self.cur_limit_gain * target_velocity_raw  # Reduce speed in sharp turns
        else:
            target_velocity = target_velocity_raw

        # ==============
        # Command Publishing
        # ==============
        # Safety check - if lookahead indices are too far from closest point
        max_index_distance = self.num_waypoints / 2  # Maximum allowable distance between indices
        steer_distance = max(steer_idx, closest_idx) - min(steer_idx, closest_idx)
        speed_distance = max(speed_idx, closest_idx) - min(speed_idx, closest_idx)
        if steer_distance > max_index_distance or speed_distance > max_index_distance:
            self.get_logger().warn("Lookahead points too far from vehicle. Emergency stop.")
            steering_angle = 0.0
            target_velocity = 0.0
        else:
            # Turn Calculation
            turn_heading = np.rad2deg(np.arctan2(turn_point[1], turn_point[0]))
            print('turn heading', turn_heading)
            if abs(turn_heading) > 30:
                if self.enable_turn:
                    self.get_logger().info(f'Turning {turn_heading} degrees')
                    self.execute_turn( (0.0, 0.0, 0.0), (0.0, 0.0, turn_heading) )

                    # turn_angle = np.rad2deg(np.arctan2(self.waypoints[turn_idx,1] - self.waypoints[turn_idx-1,1], self.waypoints[turn_idx,0] - self.waypoints[turn_idx-1,0]))
                    # start = (current_pos[0], current_pos[1], self.current_yaw)
                    # end = (self.waypoints[turn_idx,0], self.waypoints[turn_idx,1], turn_angle)

                    # print('turning from', start, 'to', end)
                    # self.execute_turn(start, end)
                else:
                    self.turn_hysteresis += 1
            else:
                self.turn_hysteresis = 0

        drive_msg = AckermannDriveStamped()
        drive_msg.header.stamp = self.get_clock().now().to_msg()
        drive_msg.drive.speed = float(np.clip(target_velocity, 0, self.base_vel))
        drive_msg.drive.steering_angle = steering_angle
        self.pub_drive.publish(drive_msg)

        # ======================
        # Visualization Markers
        # ======================
        marker_array = MarkerArray()

        # 1. Clear previous lookahead markers
        clear_marker = Marker()
        clear_marker.action = Marker.DELETEALL
        marker_array.markers.append(clear_marker)

        # 2. Add current markers with fixed IDs
        marker_array.markers.append(self.create_lookahead_marker(self.waypoints[steer_idx], (0, 0, 1), self.steer_marker_id))  # color blue
        marker_array.markers.append(self.create_lookahead_marker(self.waypoints[speed_idx], (1, 0, 0), self.speed_marker_id))  # color red

        self.pub_ld_marker.publish(marker_array)
        self.pub_path_marker.publish(self.path_marker_array)

        print(f"Lateral {lateral_displacement:.2f}, Steering: {np.rad2deg(steering_angle):.2f}, Velocity: {target_velocity:.2f}")
        print(f"Speed Heading: {np.rad2deg(speed_heading):.2f}, Brake: {brake_amount:.2f}")
        print(f"Curvature: {curvature:.2f}, Thresh?: {abs(curvature) > self.curvature_thresh}")
        print("="*20)

    def create_path_marker(self):
        """Create/replace the path marker with fixed ID"""
        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "path"
        marker.id = self.path_marker_id
        # marker.type = Marker.LINE_STRIP
        marker.type = Marker.SPHERE_LIST
        marker.action = Marker.ADD
        # marker.scale.x = 0.1
        marker.scale.x = marker.scale.y = marker.scale.z = 0.3
        marker.color.a = 1.0
        marker.color.g = 1.0  # Green
        marker.points = [Point(x=pt[0], y=pt[1], z=0.0) for pt in self.waypoints]
        return marker

    def create_lookahead_marker(self, point, color, marker_id):
        """Create lookahead marker with specified ID"""
        color = float(color[0]), float(color[1]), float(color[2])
        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = f"lookahead_{marker_id}"
        marker.id = marker_id
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.scale.x = marker.scale.y = marker.scale.z = 0.3
        marker.color.a = 1.0
        marker.color.r, marker.color.g, marker.color.b = color
        marker.pose.position.x = point[0]
        marker.pose.position.y = point[1]
        return marker


def main(args=None):
    rclpy.init(args=args)
    pure_pursuit_node = PurePursuit()
    rclpy.spin(pure_pursuit_node)
    pure_pursuit_node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
