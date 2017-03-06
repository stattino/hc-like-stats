from detectionboundary import *
import numpy as np
# SCRIPT TO RUN 22/2

m1 = 150
m2 = 50 # lol boll troll
n = 1000
dist = {1: 'norm'} #dist = {1: 'chi2', 'df': 1, 'loc': 0}
distributions = {'norm': {1: 'norm'}}
                # 'chi2': {1: 'chi2', 'df': 1, 'loc': 0}}

dense_grid = np.array([30, 30])
sparse_grid = np.array([60, 70])

hc_functions = {'CsCsHM': hc_cscshm, 'plus': hc_plus}
                #'orthodox': hc_orthodox}

for dist_type in distributions:
    dist = distributions[dist_type]
    for key in hc_functions:
        dense = dense_region(n, dense_grid, m1, m2, dist, hc_functions[key])


        msg1 = 'dense' + key
        heat_map_alt(dense, m1, msg1)
        heat_map_save(dense, m1, msg1)

        sparse = sparse_region(n, sparse_grid, m1, m2, dist, hc_functions[key])

        msg2 = 'sparse' + key
        heat_map_alt(sparse, m1, msg2)
        heat_map_save(sparse, m1, msg2)
