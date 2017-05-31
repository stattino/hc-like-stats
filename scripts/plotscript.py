from detectionboundary import *
import numpy as np

theta = 0.4

#matrix = np.loadtxt('../data/heatmap_data_HCplus_m1=10_n=100dense_worker=0_grid=50x50_time_05-30_09-55-12.txt')
#normalize_colors(matrix)

x_lim = np.array([0, 0.5])
y_lim = np.array([0, 0.5])

#heat_map_alt(matrix, 10000, x_lim, y_lim, 'DIAGNOSIS')

visualize_regions_all()

