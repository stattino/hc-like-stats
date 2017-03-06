import numpy as np
from scipy.stats import norm, chi2
from datageneration import *
from plotting import *
from highercrit import *
import matplotlib.pyplot as plt
from collections import Counter


#DEPRECATED NEEDS UPDATE
def detection_boundary(n, grids, m1, m2, dist):
    dense = dense_region(n, grids[0:2], m1, m2, dist)
    print('dense region complete')
    sparse = sparse_region(n, grids[2:4], m1, m2, dist)
    print('sparse region complete')
    heat_map_alt(dense, m1)
    heat_map_alt(sparse, m1)


def dense_region(n, grid, m1, m2, dist, hc_function):
    dense = np.zeros(grid)
    grid_x = np.linspace(0, 0.5, grid[0])
    grid_y = np.linspace(0, 0.5, grid[1])
    for beta in range(0, grid[0]):
        for r in range(0, grid[1]):
            # dense[beta, r] = error_rate(grid_x[beta], grid_y[r], m1, m2, dist)
            dense[beta, r] = compute_average_error(n, grid_x[beta], grid_y[r], m1, m2, dist, hc_function)
            print('Fraction dense region completed', r + beta*grid[1] + 1, '/', grid[0]*grid[1])
            """if beta == 0:
                dense[beta, r] = 0.6
            if r == 0:
                dense[beta, r] = 1"""
    return dense


def sparse_region(n, grid, m1, m2, dist, hc_function):
    sparse = np.zeros(grid)
    grid_x = np.linspace(0.5, 1, grid[0])
    grid_x[0] += 1/grid[0] * 1/10  # Get the sparse version of the parametrization
    grid_y = np.linspace(0, 1, grid[1])
    for beta in range(0, grid[0]):
        for r in range(0, grid[1]):
            # sparse[beta, r] = error_rate(grid_x[beta], grid_y[r], m1, m2, dist)
            sparse[beta, r] = compute_average_error(n, grid_x[beta], grid_y[r], m1, m2, dist, hc_function)
            print('Fraction of sparse region completed', r + beta * grid[1] + 1, '/', grid[0] * grid[1])
    return sparse


def compute_average_error(n, beta, r, m1, m2, dist, hc_function):
    half = int(m1/2)
    hc_type_1 = np.zeros(half)
    hc_type_2 = np.zeros(half)
    for i in range(0, half):
        x = generate_data(n, beta, r, 0, dist)
        _, hc = hc_function(x, beta, r, dist)
        hc_type_1[i] = hc
        y = generate_data(n, beta, r, 1, dist)
        _, hc = hc_function(y, beta, r, dist)
        hc_type_2[i] = hc
    d_size = 10
    deltas = 0.2*np.linspace(1, d_size, 10)
    error_sum = np.zeros(10)
    for i in range(0, 10):
        threshold = np.sqrt(2 * (1 + deltas[i]) * np.log(np.log(n)))
        error_sum[i] = sum(hc_type_1 >= threshold) + sum(hc_type_2 < threshold)
        # print('type 1: ', sum(hc_type_1 >= threshold), ' type II: ', sum(hc_type_2 < threshold))
    # print(error_sum)
    error = min(error_sum)/m1
    return error


# Old code, don't use
def error_rate(beta, r, m1, m2, dist):
    type_1_error = 0
    type_2_error = 0
    half = int(m2/2)
    for i in range(0, half):
        type_1_error += single_trial_error(m1, beta, r, 0, dist)
    for i in range(0, half):
        type_2_error += single_trial_error(m1, beta, r, 1, dist)
    average_error = (type_1_error + type_2_error) / m2
    # type_1_error = type_1_error/(m2/2)
    # type_2_error = type_2_error/(m2/2)
    return average_error  #, type_1_error, type_2_error


def single_trial_error(n, beta, r, presence, dist):
    threshold = np.sqrt(2*(1+delta)*np.log(np.log(n)))
    error = np.NaN
    if presence == 0:
        x = generate_data(n, beta, r, presence, dist)
        _, hc = hc_plus(x, beta, r, dist)
        if hc > threshold:
            error = 1
        else:
            error = 0
    elif presence == 1:
        x = generate_data(n, beta, r, presence, dist)
        _, hc = hc_plus(x, beta, r, dist)
        if hc > threshold:
            error = 0
        else:
            error = 1
    return error


def average(list):
    n = list.shape[0]
    mean = sum(list)/n
    sorted_list = list.sort()
    if n/2 % 1 != 0:
        median = sorted_list(np.ceil(n/2)) + sorted_list(np.floor(n/2)) / 2
    else:
        median = sorted_list[n/2]

    def Most_Common(lst): # BORROWED CODE
        data = Counter(lst)
        return data.most_common(1)[0][0]

    type_no = Most_Common(list)

    return mean, type_no, median