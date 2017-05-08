# from hc_tools import *
from detectionboundary import *
import numpy as np

dist = {1: 'norm'}
m1 = 2
m2 = 0 # lol boll troll
p = 10000
theta = 0.4

classification_grid = np.array([10, 10])

hc_functions = {'CsCsHM_1': hc_cscshm_1, 'CsCsHM_2': hc_cscshm_2, 'plus': hc_plus}

for key in hc_functions:
    #classification_detection_boundary(theta, p, classification_grid, m1, m2, hc_functions[key])
    error_matrix = classification_region(theta, p, classification_grid, m1, m2, hc_functions[key])
    x_lim = np.array([0, 1 - theta])
    y_lim = np.array([0, 1])
    msg = key + '_theta=' + str(theta)
    heat_map_save(error_matrix, classification_grid[0], classification_grid[1], msg)


