import cv2
import matplotlib.pyplot as plt
import numpy as np
from utils import map2px_coords, grid2map_coords, map2grid_coords, px2map_coords

class PointVisualizer:
    def __init__(self, map_file):
        self.map_file = map_file
        self.map_img = cv2.imread(self.map_file, cv2.IMREAD_GRAYSCALE)
        if self.map_img is None:
            raise FileNotFoundError(f"Could not read map image from {self.map_file}")

    def visualize_points(self, points_to_visualize_px, save_filepath=None, point_label="Points", point_type="px", show_id=False):
        
        assert point_type in ["px", "map", "grid"], "point_type must be 'px', 'map', or 'grid'"
        if point_type == "map":
            points_to_visualize_px = map2px_coords(points_to_visualize_px)
        elif point_type == "grid":
            points_to_visualize_px = grid2map_coords(points_to_visualize_px)
            points_to_visualize_px = map2px_coords(points_to_visualize_px)
        
        plt.figure()
        plt.imshow(self.map_img, cmap='gray')

        if points_to_visualize_px:
            points_x_coords = [p[0] for p in points_to_visualize_px]
            points_y_coords = [p[1] for p in points_to_visualize_px]
            plt.plot(points_x_coords, points_y_coords, 'bo', markersize=5, label=point_label)
            if show_id:
                for i, p in enumerate(points_to_visualize_px):
                    plt.text(p[0] - 1, p[1] - 1, str(i), fontsize=12, ha='right', color='red')
            title = f"Point Visualization ({len(points_to_visualize_px)} points)"
        else:
            title = f"{point_label} Visualization - No Points Provided"

        plt.legend()
        plt.title(title)
        plt.xlabel("X (pixels)")
        plt.ylabel("Y (pixels)")
        plt.axis('equal')
        plt.grid(True, linestyle='--', alpha=0.5)

        if save_filepath:
            plt.savefig(save_filepath)

        plt.show()

# Example usage (optional, for testing)
if __name__ == '__main__':
    # Create dummy files and parameters for testing
    dummy_map_file = "./maps/map.pgm"

    visualizer = PointVisualizer(
        map_file=dummy_map_file
    )
    
    start = (6, 3)
    from utils import grid2map_coords
    goal = grid2map_coords([(1169, 729)])[0]
    map_points = [start, goal]
    map_points = [[42.864115301077064, 10.843968368260533], [27.39155718331883, 5.147483022664208], [17.046962231777393, 3.3476621180896835], [17.107638309513003, 5.449165963813174], [42.81871625110311, 10.754867715841698]]
    points = map2px_coords(map_points)
    
    # grid_points = [[1170, 729], [1049, 841], [986, 925], [1005, 934], [1169, 729]]
    # map_points = grid2map_coords(grid_points)
    # points = map2px_coords(map_points)
    
    # points = [[6,3],[6,4]]
    # points = map2grid_coords(points)
    # print(points)
    
    # points = [
    #     [35.38771570222914, 9.827381412635743],
    #     [35.52221540462193, 9.871083015080627],
    #     [8.861152308636912, 3.141468271879785],
    #     [35.47681635464797, 9.78198236266179],
    #     [35.61131605704077, 9.825683965106672],
    #     [35.433114752203096, 9.91648206505458],
    #     [35.16411534741751, 9.829078860164817],
    #     [35.56761445459588, 9.960183667499463],
    #     [35.298615049810294, 9.872780462609699],
    #     [35.25321599983634, 9.783679810190863]
    # ]
    
    points = [
        [24.009127286461176, 7.319847631297742],
        [23.382027824471173, 7.190440271492169],
        [26.12421071241725, 9.047974392055162],
        [23.296322067110484, 7.68304003108938],
        [24.099925386409083, 7.498048936135415],
        [26.08050910997237, 9.182474094447953],
        [27.464115393757947, 8.814189352069098],
        [23.47112847689001, 7.145041221518214],
        [25.1708306629642, 7.176860691259588],
        [23.42742687444513, 7.279540923911005]
    ]
    
    visualizer.visualize_points(points, point_type="map", show_id=True)