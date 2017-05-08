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


"""

matrix = np.loadtxt('pdcresults/heatmap_data_CsCsHM_2_theta=0.15_grid=59x59_time_05-07_08-40-20.txt')
normalize_colors(matrix)
theta = 0.15
x_lim = np.array([0, 1-theta])
y_lim = np.array([0, 1])
p = 100
heat_map_alt(matrix, p, x_lim, y_lim, 'TESTING')
