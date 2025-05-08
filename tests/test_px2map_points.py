
import numpy as np


number_of_points = 100 
start_x = 0
start_y = 0
end_x = 1245
end_y = 824
num_points = 1000
x_coords = np.linspace(start_x, end_x, num_points)
y_coords = np.linspace(start_y, end_y, num_points)
x_y_coords = np.column_stack((x_coords, y_coords))

resolution = 0.05
img_height = 824
discrete = np.array([x_y_coords[i] for i in range(num_points)]) * resolution
x, y = discrete.T

y = img_height * resolution - y
z = np.zeros_like(x)
pts = np.vstack([x, y, z]).T
np.savetxt("points.csv", pts, delimiter=",", header="x,y,z", fmt="%.8f", comments='')