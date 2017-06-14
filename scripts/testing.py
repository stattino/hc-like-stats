from plotting import *

 # Data for testing
N = 100000
beta = 0.6
r = 0.7
signal_presence = 1
alpha_cs = 0.1
alpha_hc = 0.5
#visualize_regions_theta([0, 0.2, 0.4])
#"""
A = generate_normal_mixture(N, beta, r, signal_presence)

i_plus, hc_plus = hc_plus(A, beta, r, alpha_hc, 1)
i_cs1, crit_cs1 = hc_cscshm_1(A, beta, r, alpha_cs)
i_cs2, crit_cs2 = hc_cscshm_2(A, beta, r, alpha_cs)

#"""
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