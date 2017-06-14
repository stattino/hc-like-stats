#!/usr/bin/env python
from __future__ import division
from __future__ import print_function
from mpi4py import MPI
from cluster_heatmap import *
#import random

comm = MPI.COMM_WORLD
rank = MPI.COMM_WORLD.Get_rank()

m1 = 10
n = 100

dense_grid = np.array([50, 50])
sparse_grid = np.array([100, 100])


dense_hc, dense_cs = dense_region(n, dense_grid, m1)

heat_map_save(dense_hc, n, m1, 'TRIAL_sparse_HC')
heat_map_save(dense_cs, n, m1, 'TRIAL_sparse_CsCsHM')

sparse_hc, sparse_cs = sparse_region(n, sparse_grid, m1)

heat_map_save(sparse_hc, n, m1, 'TRIAL_sparse_HC')
heat_map_save(sparse_cs, n, m1, 'TRIAL_sparse_CsCsHM')

"""
normalize_colors(dense_hc)
normalize_colors(dense_cs)
x_lim = np.array([0, 0.5])
y_lim = np.array([0, 0.5])
heat_map_alt(dense_hc, n, x_lim, y_lim,'TRIAL_dense_hc')
heat_map_alt(dense_cs, n, x_lim, y_lim, 'TRIAL_dense_cs')

normalize_colors(sparse_hc)
normalize_colors(sparse_cs)
x_lim = np.array([0.5, 1])
y_lim = np.array([0, 1])
heat_map_alt(sparse_hc, n, x_lim, y_lim, 'TRIAL_sparse_hc')
heat_map_alt(sparse_cs, n, x_lim, y_lim, 'TRIAL_sparse_cs')
"""