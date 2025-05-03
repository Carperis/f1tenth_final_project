import cv2
import numpy as np
import matplotlib.pyplot as plt

# Map metadata from YAML
resolution = 0.05  # meters/pixel
origin = np.array([0.0, 0.0])  # meters (X, Y)

# Load the map
map_img = cv2.imread("map.pgm", cv2.IMREAD_GRAYSCALE)
height, width = map_img.shape

clicked_points = []

def onclick(event):
    if event.xdata is not None and event.ydata is not None:
        x_px, y_px = int(event.xdata), int(event.ydata)
        # Convert pixel coordinates to world (real-world meters)
        x_m = origin[0] + x_px * resolution
        y_m = origin[1] + (height - y_px) * resolution  # y-axis flip
        clicked_points.append((x_m, y_m))
        print(f"Clicked point (world coords): ({x_m:.2f}, {y_m:.2f})")

        if len(clicked_points) == 2:
            plt.close()

# Show map for clicking
fig, ax = plt.subplots()
ax.imshow(map_img, cmap='gray')
ax.set_title("Click START point and then END point")
cid = fig.canvas.mpl_connect('button_press_event', onclick)
plt.show()

# Save results
if len(clicked_points) == 2:
    start, goal = clicked_points
    np.savetxt("start_goal.csv", np.array([start, goal]), delimiter=';', fmt="%.4f", header="x_m;y_m", comments="# ")
    print("\n Saved to start_goal.csv")
else:
    print("You did not click two points.")
