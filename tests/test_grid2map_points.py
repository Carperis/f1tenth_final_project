from utils import grid2map_coords, map2grid_coords

grid_points =  [(1000, 1000), (1169, 729), (1005, 934), (986, 925), (1049, 841), (1170, 729)]
# grid_points = [(1055, 881), (1053, 889), (1078, 870), (1049, 886), (1053, 881)]

map_points = grid2map_coords(grid_points)
print("Map points:", map_points)

grid_points = map2grid_coords(map_points)
print("Grid points:", grid_points)