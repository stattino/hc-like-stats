import numpy as np
from scipy.stats import norm, chi2
from datageneration import *
from plotting import *
from highercrit import *
import matplotlib.pyplot as plt
from collections import Counter


def classification_detection_boundary(theta, p, grid, m1, m2, hc_function):
    error_matrix = classification_region(theta, p, grid, m1, m2, hc_function)
    x_lim = np.array([0, 1-theta])
    y_lim = np.array([0, 1-theta])
    normalize_colors(error_matrix)
    heat_map_alt(error_matrix, p, x_lim, y_lim, 'classification')


def classification_region(theta, p, grid, m1, m2, hc_function):
    dense = np.zeros(grid)
    grid_x = np.linspace(0, 1-theta, grid[0])
    grid_y = np.linspace(0, 1-theta, grid[1])
    for beta in range(0, grid[0]):
        for r in range(0, grid[1]):
            # dense[beta, r] = error_rate(grid_x[beta], grid_y[r], m1, m2, dist)
            dense[beta, r] = classification_error(theta, grid_x[beta], grid_y[r], p, m1, m2, hc_function)
            print('Fraction dense region completed', r + beta*grid[1] + 1, '/', grid[0]*grid[1])
    return dense


def classification_error(theta, beta, r, p, m1, m2, hc_function):
    error = 0
    for i in range(0, m1):
        x_train, y_train, x_test, y_test = generate_classification_data(p, theta, beta, r)
        _, weights = hc_thresholding(x_train, y_train, beta, r, hc_function, 'clip')
        y_attempt = discriminant_rule(weights, x_test)
        error += sum(y_attempt != y_test) / y_test.shape
    return error/m1


def testing_detection_boundary(n, grids, m1, m2, dist, hc_function):

    dense = dense_region(n, grids[0:2], m1, m2, dist, hc_function)
    print('dense region complete')
    x_lim = np.array([0, 0.5])
    y_lim = np.array([0, 0.5])
    normalize_colors(dense)
    heat_map_alt(dense, m1, x_lim, y_lim, 'dense')

    sparse = sparse_region(n, grids[2:4], m1, m2, dist, hc_function)
    print('sparse region complete')
    x_lim = np.array([0.5, 1])
    y_lim = np.array([0, 1])
    normalize_colors(sparse)
    heat_map_alt(sparse, m1, x_lim, y_lim,  'sparse')


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


def normalize_colors(matrix):
    n, p = matrix.shape
    for i in range(0, n):
        for j in range(0, p):
            if matrix[i, j] > 0.5:
                matrix[i, j] = 0.5
    if n*p > 1000:
        index = matrix.argmin()
        matrix[index] = 0
    index = matrix.argmax()
    index_x = index%p
    index_y = int((index-index_x) /p)
    matrix[index_x, index_y] = 0.5
