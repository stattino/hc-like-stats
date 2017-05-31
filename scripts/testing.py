import numpy as np
from datageneration import *
from highercrit import *
from datageneration import *
from plotting import *
from detectionboundary import *

 # Data for testing
N = 1000
beta = 0.8
r = 0.4
signal_presence = 1

df = 1
location = 0
dist = {1: 'norm'}
# dist = {1: 'chi2', 'df': 1, 'loc': 0}
"""
A = generate_data(N, beta, r, signal_presence, dist)
#A = generate_normal_mixture(N, beta, r)
#A = generate_chi2_mixture(N, beta, r, df, location)
plot_option = 1

i, hc = hc_orthodox(A, beta, r, dist, plot_option)
i_plus, hc_plus = hc_plus(A, beta, r, dist, plot_option)
i_cs, hc_cs = hc_cscshm(A, beta, r, dist, plot_option)

matrix = np.loadtxt('../data/heatmap_data_CsCsHM_1_m1=10_n=100_grid=10x10_time_05-25_11-21-08.txt')
normalize_colors(matrix)
theta = 0.15
x_lim = np.array([0, 1-theta])
y_lim = np.array([0, 1])
p = 100
"""
n = 100

#heat_map_alt(matrix, p, x_lim, y_lim, 'TESTING')
dense_error_matrix = np.loadtxt('../data/heatmap_data_CsCsHM2_m1=10_n=100_grid=35x35_time_05-25_11-39-46.txt')
normalize_colors(dense_error_matrix)
x_lim = np.array([0, 0.5])
y_lim = np.array([0, 0.5])
heat_map_alt(dense_error_matrix, n, x_lim, y_lim, 'TESTING')

sparse_error_matrix = np.loadtxt('../data/heatmap_data_CsCsHM2_m1=10_n=100_grid=70x70_time_05-25_11-39-46.txt')
normalize_colors(sparse_error_matrix)
x_lim = np.array([0.5, 1])
y_lim = np.array([0, 1])
heat_map_alt(sparse_error_matrix, n, x_lim, y_lim, 'test_test')




"""
dense_error_matrix = np.loadtxt('../data/heatmap_data_CsCsHM_1_m1=10_n=100_grid=10x10_time_05-25_11-21-08.txt')
normalize_colors(dense_error_matrix)
x_lim = np.array([0, 0.5])
y_lim = np.array([0, 0.5])
heat_map_alt(dense_error_matrix, n, x_lim, y_lim, 'TESTING')

sparse_error_matrix = np.loadtxt('../data/heatmap_data_CsCsHM_1_m1=10_n=100_grid=10x10_time_05-25_11-21-08.txt')
normalize_colors(sparse_error_matrix)
x_lim = np.array([0.5, 1])
y_lim = np.array([0, 1])
heat_map_alt(sparse_error_matrix, n, x_lim, y_lim, 'test_test')
"""