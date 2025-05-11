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

    def visualize_points(self, points_data, save_filepath=None, point_label="", point_type="px", show_id=False, point_color='bo'):
        """
        Visualizes a single class of points on the map.
        points_data: List of points to visualize.
        save_filepath: Path to save the visualization (optional).
        point_label: Label for this class of points (optional).
        point_type: Type of the input points ('px', 'map', or 'grid').
        show_id: Whether to show the ID of each point.
        point_color: Color and marker style for the points (e.g., 'bo', 'rx', 'g^'). Defaults to 'bo'.
        """
        assert point_type in ["px", "map", "grid"], "point_type must be 'px', 'map', or 'grid'"
        
        points_to_visualize_px = [] # Initialize to ensure it's defined
        if point_type == "px":
            points_to_visualize_px = list(points_data) # Make a copy if it's already px
        elif point_type == "map":
            points_to_visualize_px = map2px_coords(points_data)
        elif point_type == "grid":
            # Ensure points_data is transformed correctly if it's grid type
            map_coords = grid2map_coords(points_data)
            points_to_visualize_px = map2px_coords(map_coords)
        
        plt.figure()
        plt.imshow(self.map_img, cmap='gray')

        if points_to_visualize_px:
            points_x_coords = [p[0] for p in points_to_visualize_px]
            points_y_coords = [p[1] for p in points_to_visualize_px]
            plt.plot(points_x_coords, points_y_coords, point_color, markersize=5, label=point_label) # Use point_color
            if show_id:
                for i, p in enumerate(points_to_visualize_px):
                    # Use the color part of point_color for text, default to 'b' if not a valid color string start
                    text_color = point_color[0] if isinstance(point_color, str) and len(point_color) > 0 else 'b'
                    plt.text(p[0] - 1, p[1] - 1, str(i), fontsize=12, ha='right', color=text_color)
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

    def visualize_multiclass_points(self, points_data, save_filepath=None, point_type="px", show_id=False, point_labels=[], point_colors=[]):
        """
        Visualizes multiple classes of points on the map.
        points_data: A list of lists of points. Each inner list represents a class of points.
        save_filepath: Path to save the visualization (optional).
        point_type: Type of the input points ('px', 'map', or 'grid').
        show_id: Whether to show the ID of each point.
        point_labels: An optional list of strings. Provides labels for each corresponding list of points in points_data.
                      If shorter than points_data, generic labels (e.g., "Class N") are used for the remaining classes. Defaults to [].
        point_colors: An optional list of strings (e.g., ['bo', 'rx', 'g^']). Provides colors/markers for each corresponding
                      list of points in points_data. If shorter than points_data, a default style ('bo') is used
                      for the remaining classes. Defaults to [].
        """
        assert point_type in ["px", "map", "grid"], "point_type must be 'px', 'map', or 'grid'"

        plt.figure()
        plt.imshow(self.map_img, cmap='gray')
        
        total_points = 0

        for idx, current_points_list in enumerate(points_data):
            points_to_visualize = current_points_list # current_points_list is a list of points for the current class

            # Determine label for the current class
            if idx < len(point_labels):
                label_to_use = point_labels[idx]
            else:
                label_to_use = f"Class {idx + 1}" # Fallback to generic label
            
            # Determine color and marker for the current class
            if idx < len(point_colors):
                color_marker_to_use = point_colors[idx]
            else:
                color_marker_to_use = 'bo' # Default to blue circles

            if not points_to_visualize:
                continue

            points_to_visualize_px_class = list(points_to_visualize) # Make a copy

            if point_type == "map":
                points_to_visualize_px_class = map2px_coords(points_to_visualize_px_class)
            elif point_type == "grid":
                points_to_visualize_px_class = grid2map_coords(points_to_visualize_px_class)
                points_to_visualize_px_class = map2px_coords(points_to_visualize_px_class)
            
            if points_to_visualize_px_class:
                points_x_coords = [p[0] for p in points_to_visualize_px_class]
                points_y_coords = [p[1] for p in points_to_visualize_px_class]
                plt.plot(points_x_coords, points_y_coords, color_marker_to_use, markersize=5, label=label_to_use)
                total_points += len(points_to_visualize_px_class)
                if show_id:
                    for i, p in enumerate(points_to_visualize_px_class):
                        # Use the color part of color_marker_to_use for text, default to 'b' if not a valid color string start
                        text_color = color_marker_to_use[0] if isinstance(color_marker_to_use, str) and len(color_marker_to_use) > 0 else 'b'
                        plt.text(p[0] - 1, p[1] - 1, str(i), fontsize=12, ha='right', color=text_color)

        if total_points > 0:
            title = f"Multiclass Point Visualization ({total_points} points)"
        else:
            title = "Multiclass Point Visualization - No Points Provided"

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
    
    points1 = [
        [35.38771570222914, 9.827381412635743],
        [35.52221540462193, 9.871083015080627],
        [8.861152308636912, 3.141468271879785],
        [35.47681635464797, 9.78198236266179],
        [35.61131605704077, 9.825683965106672],
        [35.433114752203096, 9.91648206505458],
        [35.16411534741751, 9.829078860164817],
        [35.56761445459588, 9.960183667499463],
        [35.298615049810294, 9.872780462609699],
        [35.25321599983634, 9.783679810190863]
    ]
    
    points2 = [
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
    
    visualizer.visualize_points(points1, point_type="map", show_id=True, point_color='ro')
    
    # Updated example for visualize_multiclass_points
    all_points_data = [points1, points2] # points1 and points2 are defined above as lists of points
    example_labels = ["Path Alpha", "Path Beta"]
    example_colors = ['ro', 'bo'] # Example: red circles with line, blue squares with dashed line
    
    visualizer.visualize_multiclass_points(
        all_points_data, 
        point_type="map", 
        # show_id=True, 
        point_labels=example_labels, 
        point_colors=example_colors
    )