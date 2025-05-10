from udp_comm import UDPComm
from a_star_planner import AStarPlanner
import numpy as np
import time

class RobotController:
    def __init__(self, goals, planner, comm,
                 goal_reach_threshold=1.0,
                 wait_time_get_pos=2.0):
        self.goals = [np.array(g) for g in goals]
        self.current_car_position = None
        self.planner = planner
        self.comm = comm
        self.goal_reach_threshold = goal_reach_threshold
        self.wait_time_get_pos = wait_time_get_pos

    def _get_current_position_with_retry(self, context_message=""):
        """Helper method to get current car position with retries."""
        position = None
        while position is None:
            time.sleep(self.wait_time_get_pos)
            raw_position = self.comm.get_car_position()
            if raw_position is not None:
                position = np.array(raw_position)
            else:
                print(f"Position not received {context_message}. Retrying...")
        return position

    def _wait_for_goal_arrival(self, current_goal):
        print(f"Waiting for car to reach goal: {current_goal}")
        while True:
            self.current_car_position = self._get_current_position_with_retry(context_message=f"while navigating to {current_goal}")
            distance_to_goal = np.linalg.norm(self.current_car_position - current_goal)
            if distance_to_goal < self.goal_reach_threshold:
                print(f"Car reached goal: {current_goal}.")
                break
            print(f"Car Pos: {self.current_car_position}, Goal: {current_goal}, Dist: {distance_to_goal:.2f}m", end="\r")
        return self.current_car_position

    def run(self):
        input("Position car at start and press Enter to begin...")

        print("Getting initial car position...")
        active_start_position = self._get_current_position_with_retry(context_message="for initial position")

        for i, goal_pos in enumerate(self.goals):
            goal_processed_successfully = False
            print(f"\nProcessing Goal {i+1}/{len(self.goals)}: {goal_pos}")

            while not goal_processed_successfully:
                print(f"Planning path from {active_start_position} to {goal_pos}...")
                _, smoothed_path = self.planner.plan(active_start_position, goal_pos, near=True)

                if smoothed_path:
                    print(f"Path found ({len(smoothed_path)} points). Sending to car...")
                    
                    send_success = False
                    while not send_success:
                        send_success = self.comm.send_trajectory_from_list(smoothed_path, is_grid=False)
                        if send_success:
                            print(f"Path to goal {goal_pos} sent.")
                            self.current_car_position = self._wait_for_goal_arrival(goal_pos)
                            active_start_position = self.current_car_position
                            goal_processed_successfully = True
                        else:
                            time.sleep(1)
                
                else:
                    print("Path planning failed. Retrying after getting new car position.")
                    print("Ensure car is in a clear area or adjust its position.")
                    active_start_position = self._get_current_position_with_retry(context_message="after path planning failure")
        
        print("\nAll goals processed.")
        self.comm.close_socket()

if __name__ == "__main__":    
    target_goals = [
        [12.45, 2.96],
        [35.25, 8.81],
        [50.9, 18.36],
    ]
    map_pgm_file = "./maps/map.pgm"
    map_config_yaml_file = "./maps/map.yaml"
    vehicle_clearance_m = 0.4
    jetson_udp_ip = "10.103.74.128"

    planner = AStarPlanner(
        map_file=map_pgm_file,
        output_path_file="path.csv",
        map_yaml_file=map_config_yaml_file,
        clearance_radius_m=vehicle_clearance_m
    )

    comm = UDPComm(
        udp_ip=jetson_udp_ip,
    )

    controller = RobotController(
        goals=target_goals,
        planner=planner,
        comm=comm,
        goal_reach_threshold=1.0,
        wait_time_get_pos=2.0
    )
    controller.run()

