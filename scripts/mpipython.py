
from detectionboundary import *


dist = {1: 'norm'}
m1 = 10
m2 = 0 # lol boll troll
n = 100

dense_grid = np.array([35, 35])
sparse_grid = np.array([70, 70])

hc_functions = {'HCplus': hc_plus} #, 'CsCsHM2': hc_cscshm_2,'CsCsHM1': hc_cscshm_1 }

for key in hc_functions:
    dense_error_matrix = dense_region(n, dense_grid, m1, m2, dist, hc_functions[key])
    sparse_error_matrix = sparse_region(n, sparse_grid, m1, m2, dist, hc_functions[key])

    msg_1 = key + '_m1=' + str(m1) + '_n=' + str(n) + 'dense'
    msg_2 = key + '_m1=' + str(m1) + '_n=' + str(n) + 'sparse'

    heat_map_save(dense_error_matrix , dense_grid[0], dense_grid[1], msg_1)
    heat_map_save(sparse_error_matrix, sparse_grid[0], sparse_grid[1], msg_2)
