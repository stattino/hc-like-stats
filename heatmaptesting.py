from detectionboundary import *
import numpy as np

grid = np.array([10, 10, 5, 10])
m1 = 80
m2 = 50 # lol boll troll
n = 1000
#dist = {1: 'norm'}
dist = {1: 'chi2', 'df': 1, 'loc': 0}

#detection_boundary(n, grid, m1, m2, dist)



dense_grid = np.array([10, 10])
sparse_grid = np.array([5, 10])

#dense = dense_region(n, dense_grid, m1, m2, dist)
#heat_map_alt(dense, m1)
#heat_map_save(dense, m1, 'dense')

sparse = sparse_region(n, sparse_grid, m1, m2, dist, hc_cscshm)
heat_map_alt(sparse, m1)
heat_map_save(sparse, m1, 'sparse')
