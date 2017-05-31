from detectionboundary import *
import numpy as np


dist = {1: 'norm'}
m1 = 100
m2 = 0 # lol boll troll
n = 10000

dense_grid = np.array([35, 35])
sparse_grid = np.array([70, 70])

hc_functions = {'CsCsHM1': hc_cscshm_1, 'CsCsHM2': hc_cscshm_2, 'HCplus': hc_plus}

for key in hc_functions:
    dense_error_matrix = dense_region(n, dense_grid, m1, m2, dist, hc_functions[key])
    sparse_error_matrix = sparse_region(n, sparse_grid, m1, m2, dist, hc_functions[key])

    msg_1 = key + '_m1=' + str(m1) + '_n=' + str(n)
    msg_2 = key + '_m1=' + str(m1) + '_n=' + str(n)

    heat_map_save(dense_error_matrix , dense_grid[0], dense_grid[1], msg_1)
    heat_map_save(sparse_error_matrix, sparse_grid[0], sparse_grid[1], msg_2)

#comm = MPI.COMM_WORLD

#print("Hello! I'm rank %d from %d running in total..." % (comm.rank, comm.size))

#comm.Barrier()   # wait for everybody to synchronize _her