from detectionboundary import *
import numpy as np

theta = 0.4

matrix = np.loadtxt('../data/heatmap_data_CsCsHM_2_theta=0.4_grid=10x10_time_05-08_15-43-41.txt')
normalize_colors(matrix)

x_lim = np.array([0, 1-theta])
y_lim = np.array([0, 1])

heat_map_alt(matrix, 10000, x_lim, y_lim, 'DIAGNOSIS')
