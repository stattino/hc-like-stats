from heatmapthresholding import *
from detectionboundary import normalize_colors, heat_map_save
from plotting import heat_map_alt
import os

m1 = 3

p = 1000
theta = 0.4
hc = 2

"""
alpha = 1
t_type = 'hc'
#"""
alpha = 0.2
t_type = 'cs'
#"""

#"""
classification_grid = np.array([30, 30])


error_matrix = classification_region(theta, p, classification_grid, m1, hc, alpha, t_type)
normalize_colors(error_matrix)

x_lim = np.array([0, 1 - theta])
y_lim = np.array([0, 1])

msg = '_' + t_type + '_theta=' + str(theta) + '_m1=' + str(m1) + '_p=' + str(p)
#heat_map_save(error_matrix, classification_grid[0], classification_grid[1], msg)
heat_map_alt(error_matrix, p, x_lim, y_lim, msg)
#"""

"""
x_lim = np.array([0, 1 - theta])
y_lim = np.array([0, 1])
for fn in os.listdir('../src/'):
    if fn.endswith('.txt') and fn.startswith('heatmap_data__cs'):
        print(fn)
        mat = np.loadtxt('../src/' + fn)
        heat_map_alt(mat, p, x_lim, y_lim, fn)
#"""

