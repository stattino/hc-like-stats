from detectionboundary import *
import numpy as np

grid = np.array([10, 10, 5, 10])
m1 = 80
m2 = 50 # lol boll troll
n = 1000
p = 1000
theta = 0.15
dist = {1: 'norm'}
#dist = {1: 'chi2', 'df': 1, 'loc': 0}

dense_grid = np.array([20, 20])
sparse_grid = np.array([20, 20])

grids = np.append(dense_grid, sparse_grid)

classification_grid = np.array([50, 50])

classification_detection_boundary(theta, p, classification_grid, m1, m2, hc_cscshm_2)

#testing_detection_boundary(n, grids, m1, m2, dist, hc_cscshm)