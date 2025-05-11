from udp_comm import UDPComm
from a_star_planner import AStarPlanner
import numpy as np
import time

class RobotController:
    def __init__(self, goals, planner, comm,
                 goal_reach_threshold=1.0,
                 retry_wait_time=2.0):
        self.goals = [np.array(g) for g in goals]
        self.current_pos = None
        self.planner = planner
        self.comm = comm
        self.goal_reach_threshold = goal_reach_threshold
        self.retry_wait_time = retry_wait_time

    def _get_car_pos(self, context_message=""):
        """Helper method to get current car position with retries."""
        position = None
        while position is None:
            time.sleep(self.retry_wait_time)
            raw_position = self.comm.get_car_position()
            if raw_position is not None:
                position = np.array(raw_position)
            else:
                print(f"Position not received {context_message}. Retrying...")
        return position

    def _check_goal_reached(self, goal):
        print(f"Waiting for car to reach goal: {goal}")
        while True:
            self.current_pos = self._get_car_pos(context_message=f"while navigating to {goal}")
            distance_to_goal = np.linalg.norm(self.current_pos - goal)
            if distance_to_goal < self.goal_reach_threshold:
                print(f"Car reached goal: {goal}.")
                break
            print(f"Car Pos: {self.current_pos}, Goal: {goal}, Dist: {distance_to_goal:.2f}m", end="\r")
        return self.current_pos

    def run(self):
        input("Position car at start and press Enter to begin...")

        print("Getting initial car position...")
        current_pos = self._get_car_pos(context_message="for initial position")

        for i, goal in enumerate(self.goals):
            goal_reached = False
            print(f"\nProcessing Goal {i+1}/{len(self.goals)}: {goal}")

            while not goal_reached:
                print(f"Planning path from {current_pos} to {goal}...")
                _, path = self.planner.plan(current_pos, goal, near=True)

                if path:
                    print(f"Path found ({len(path)} points). Sending to car...")
                    
                    path_sent = False
                    while not path_sent:
                        path_sent = self.comm.send_trajectory_from_list(path)
                        if path_sent:
                            print(f"Path to goal {goal} sent.")
                            self.current_pos = self._check_goal_reached(goal)
                            current_pos = self.current_pos
                            goal_reached = True

                else:
                    print("Path planning failed. Retrying after getting new car position.")
                    print("Ensure car is in a clear area or adjust its position.")
                    current_pos = self._get_car_pos(context_message="after path planning failure")
        
        print("\nAll goals processed.")
        self.comm.close_socket()

if __name__ == "__main__": 
    
    from viz_map_points import PointVisualizer
    viz = PointVisualizer(
        map_file="./maps/map.pgm"
    )
    
    goals = [
        [12.45, 2.96],
        [35.25, 8.81],
        [50.9, 18.36],
    ]
    goals = np.load("goals.npy").tolist()
    print(f"Goals loaded: {goals}")
    viz.visualize_points(goals, point_label="Goals", point_type="map", show_id=True)
    
    pgm_file = "./maps/map.pgm"
    yaml_file = "./maps/map.yaml"
    clearance = 0.4
    udp_ip = "10.103.87.204"

    planner = AStarPlanner(
        map_file=pgm_file,
        output_path_file="path.csv",
        map_yaml_file=yaml_file,
        clearance_radius_m=clearance
    )

    comm = UDPComm(
        udp_ip=udp_ip,
        timeout=300,
    )

    controller = RobotController(
        goals=goals,
        planner=planner,
        comm=comm,
        goal_reach_threshold=2.0,
        retry_wait_time=2.0
    )
    controller.run()

