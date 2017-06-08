import numpy as np
from scipy.stats import norm
from plotting import *
from detectionboundary import normalize_colors

def generate_normal_mixture(n, beta, r, signal_presence):
    epsilon = np.power(n, -beta)
    if beta > 0.5:
        mu0 = np.power(2*r*np.log(n), 0.5)
    else:
        mu0 = np.power(n, -r)
    x = np.random.normal(0, 1, n)
    if signal_presence == 1:
        n_signals = int(np.ceil(epsilon*n))
        n = int(n)
        index = np.random.choice(n, n_signals, False)
        x[index] = np.random.normal(mu0, 1, n_signals)
    return x


def hc_cscshm_1(x, beta, r):
    #  Csörgós Csörgós Horvath Mason
    n = x.shape[0]
    pi = calculate_p_values(x)
    sorted_pi, _ = sort_by_size(pi)
    hc_vector = np.zeros(n)
    n_float = float(n)
    alpha_0 = 1.0
    trunc = np.floor(alpha_0 * n).astype(int)
    for i in range(1, trunc):
        if sorted_pi[i] > 1/n:
            pi_val = sorted_pi[i]
            norm = pi_val * (1 - pi_val)
            hc_vector[i] = np.sqrt(n_float) * (i/n_float - pi_val) / np.sqrt(norm * np.log(np.log(1 / norm)))
    i_opt = np.argmax(hc_vector)
    hc_opt = np.max(hc_vector)
    #print('Optimal HC:', hc_opt, 'index i_opt:', i_opt)
    return i_opt, hc_opt


def hc_cscshm_2(x, beta, r):
    #  Csörgós Csörgós Horvath Mason
    n = x.shape[0]
    pi = calculate_p_values(x)
    sorted_pi, _ = sort_by_size(pi)
    hc_vector = np.zeros(n)
    n_float = float(n)
    alpha_0 = 0.5
    trunc = np.floor(alpha_0 * n).astype(int)
    for i in range(1, trunc):
        #if sorted_pi[i] > 1/np.power(n, 2):
            pi_val = sorted_pi[i]
            norm = pi_val * (1 - pi_val)
            hc_vector[i] = np.sqrt(n_float) * (i/n_float - pi_val) / (np.sqrt(norm) * np.log(np.log(1 / norm)))
    i_opt = np.argmax(hc_vector)
    hc_opt = np.max(hc_vector)
    #print('Optimal HC:', hc_opt, 'index i_opt:', i_opt)
    return i_opt, hc_opt


def hc_plus(x, beta, r):
    n = x.shape[0]
    pi = calculate_p_values(x)
    sorted_pi, _ = sort_by_size(pi)
    hc_vector = np.zeros(n)
    n_float = float(n)
    alpha_0 = 0.5
    trunc = np.floor(alpha_0 * n).astype(int)
    for i in range(1, trunc):
        if sorted_pi[i] > 1/n:
            hc_vector[i] = np.sqrt(n_float) * (i/n_float - sorted_pi[i]) / np.sqrt(sorted_pi[i]*(1 - sorted_pi[i]))


    i_opt = np.argmax(hc_vector)
    hc_opt = np.max(hc_vector)
    #print('Optimal HC:', hc_opt, 'index i_opt:', i_opt)
    return i_opt, hc_opt


def calculate_p_values(x):
    p_values = norm.sf(x)
    return p_values


def sort_by_size(pi):
    sort_index = np.argsort(pi)
    pi = np.sort(pi)
    return pi, sort_index


def dense_region(n, grid, m1):
    dense_cs = np.zeros(grid)
    dense_hc = np.zeros(grid)

    grid_x = np.linspace(0, 0.5, grid[0])
    grid_y = np.linspace(0, 0.5, grid[1])
    for beta in range(0, grid[0]):
        for r in range(0, grid[1]):
            # dense[beta, r] = error_rate(grid_x[beta], grid_y[r], m1, m2, dist)
            dense_hc[beta, r], dense_cs[beta, r] = compute_average_error(n, grid_x[beta], grid_y[r], m1)
            if ( (r + beta*grid[1] + 1) % (grid[0]*grid[1]/10) == 0):
                print('Fraction of dense region completed', 100*(r + beta*grid[1] + 1) / (grid[0]*grid[1]) , '%')
    return dense_hc, dense_cs


def sparse_region(n, grid, m1):
    sparse_hc = np.zeros(grid)
    sparse_cs = np.zeros(grid)

    grid_x = np.linspace(0.5, 1, grid[0])
    grid_x[0] += 1/grid[0] * 1/10  # Get the sparse version of the parametrization
    grid_y = np.linspace(0, 1, grid[1])
    for beta in range(0, grid[0]):
        for r in range(0, grid[1]):
            # sparse[beta, r] = error_rate(grid_x[beta], grid_y[r], m1, m2, dist)
            if (grid_x[beta] < 0.75 and grid_y[r] < grid_x[beta]-0.5):
                sparse_hc[beta, r] = 0.5
                sparse_cs[beta, r] = 0.5
            elif (grid_x[beta] >= 0.75 and grid_y[r] < np.power(1-np.sqrt(1-grid_x[beta]), 2)):
                sparse_hc[beta, r] = 0.5
                sparse_cs[beta, r] = 0.5
            else:
                sparse_hc[beta, r], sparse_cs[beta, r] = compute_average_error(n, grid_x[beta], grid_y[r], m1)
            if ((r + beta * grid[1] + 1) % (grid[0] * grid[1] / 10) == 0):
                print('Fraction of sparse region completed', 100 * (r + beta * grid[1] + 1) / (grid[0] * grid[1]), '%')
            #print('Fraction of sparse region completed', r + beta * grid[1] + 1, '/', grid[0] * grid[1])
    return sparse_hc, sparse_cs


def compute_average_error(n, beta, r, m1):
    half = int(m1/2)
    hc_type_1_hc = np.zeros(half)
    hc_type_2_hc = np.zeros(half)

    hc_type_1_CS = np.zeros(half)
    hc_type_2_CS = np.zeros(half)
    for i in range(0, half):
        x = generate_normal_mixture(n, beta, r, 0)
        _, hc = hc_plus(x, beta, r)
        hc_type_1_hc[i] = hc
        _, hc = hc_cscshm_2(x, beta, r)
        hc_type_1_CS[i] = hc

        y = generate_normal_mixture(n, beta, r, 1)
        _, hc = hc_plus(y, beta, r)
        hc_type_2_hc[i] = hc
        #print('HC_plus: ', hc)
        _, hc = hc_cscshm_2(y, beta, r)
        hc_type_2_CS[i] = hc
        #print('Ccshm: ', hc)
    d_size = 10
    deltas = 0.1*np.linspace(1, d_size, 10)
    error_sum_hc = np.zeros(10)
    error_sum_CS = np.zeros(10)
    print(beta, r)
    for i in range(0, 10):
        threshold = np.sqrt(2 * (1 + deltas[i]) * np.log(np.log(n)))
        threshold_CS = 1.86*np.sqrt(1 + 2*deltas[i])
        error_sum_hc[i] = sum(hc_type_1_hc >= threshold) + sum(hc_type_2_hc < threshold)
        error_sum_CS[i] = sum(hc_type_1_CS >= threshold_CS) + sum(hc_type_2_CS < threshold_CS)
        print('HC type 1:     ', sum(hc_type_1_hc >= threshold), ' type II: ', sum(hc_type_2_hc < threshold))
        print('CsCsHM type 1: ', sum(hc_type_1_CS >= threshold_CS), ' type II: ', sum(hc_type_2_CS < threshold_CS))

    # print(error_sum)
    error_hc = min(error_sum_hc)/m1
    error_cs = min(error_sum_CS)/m1
    return error_hc, error_cs

def find_thresholds(n, beta, r, m1, m2):
    chosen_threshold_cs = np.zeros(m2)
    chosen_threshold_hc = np.zeros(m2)
    for j in range(0, m2):
        print(j, '/', m2)
        half = int(m1 / 2)
        hc_type_1_hc = np.zeros(half)
        hc_type_2_hc = np.zeros(half)
        hc_type_1_CS = np.zeros(half)
        hc_type_2_CS = np.zeros(half)
        for i in range(0, half):
            x = generate_normal_mixture(n, beta, r, 0)
            _, hc = hc_plus(x, beta, r)
            hc_type_1_hc[i] = hc
            _, hc = hc_cscshm_2(x, beta, r)
            hc_type_1_CS[i] = hc

            y = generate_normal_mixture(n, beta, r, 1)
            _, hc = hc_plus(y, beta, r)
            hc_type_2_hc[i] = hc
            # print('HC_plus: ', hc)
            _, hc = hc_cscshm_2(y, beta, r)
            hc_type_2_CS[i] = hc
            # print('Ccshm: ', hc)
        d_size = 10
        deltas = 0.2 * np.linspace(1, d_size, 10)
        error_sum_hc = np.zeros(10)
        error_sum_CS = np.zeros(10)
        for i in range(0, 10):
            threshold = np.sqrt(2 * (1 + deltas[i]) * np.log(np.log(n)))
            threshold_CS = 1.86 * np.sqrt(1 + 2 * deltas[i])
            error_sum_hc[i] = sum(hc_type_1_hc >= threshold) + sum(hc_type_2_hc < threshold)
            error_sum_CS[i] = sum(hc_type_1_CS >= threshold_CS) + sum(hc_type_2_CS < threshold_CS)
        # print(error_sum)
        error_hc = min(error_sum_hc) / m1
        error_cs = min(error_sum_CS) / m1

        i_opt_cs = np.argmin(error_sum_CS)
        i_opt_hc = np.argmin(error_sum_hc)
        chosen_threshold_hc[j] = np.sqrt(2 * (1 + deltas[i_opt_hc]) * np.log(np.log(n)))
        chosen_threshold_cs[j] = 1.86 * np.sqrt(1 + 2*deltas[i_opt_cs])

    return chosen_threshold_hc, chosen_threshold_cs


def find_HCs(n, beta, r, m1, m2):
    half = int(m1 / 2)
    critical_hc = np.zeros((m2*half, 2))
    critical_CS = np.zeros((m2*half, 2))
    for j in range(0, m2):
        print(j+1, '/', m2)
        hc_type_1_hc = np.zeros(half)
        hc_type_2_hc = np.zeros(half)
        hc_type_1_CS = np.zeros(half)
        hc_type_2_CS = np.zeros(half)
        for i in range(0, half):
            print(i+1, '/', half)
            x = generate_normal_mixture(n, beta, r, 0)
            _, hc_1 = hc_plus(x, beta, r)
            hc_type_1_hc[i] = hc_1
            _, cs_1 = hc_cscshm_2(x, beta, r)
            hc_type_1_CS[i] = cs_1
            # Fill the histos
            critical_hc[j*half + i, 0] = hc_1
            critical_CS[j*half + i, 0] = cs_1

            y = generate_normal_mixture(n, beta, r, 1)
            _, hc_2 = hc_plus(y, beta, r)
            hc_type_2_hc[i] = hc_2
            # print('HC_plus: ', hc)
            _, cs_2 = hc_cscshm_2(y, beta, r)
            hc_type_2_CS[i] = cs_2
            # print('Ccshm: ', hc)
            critical_hc[j*(half) + i, 1] = hc_2
            critical_CS[j*(half) + i, 1] = cs_2

    return critical_hc, critical_CS

# Testing of the functions

n = 500
m1 = 100
dense_grid = np.array( [10, 10])
"""
dense_hc, dense_cs = dense_region(n, dense_grid, m1)

normalize_colors(dense_hc)
normalize_colors(dense_cs)
x_lim = np.array([0, 0.5])
y_lim = np.array([0, 0.5])
heat_map_alt(dense_hc, n, x_lim, y_lim,'TRIAL_dense_hc')
heat_map_alt(dense_cs, n, x_lim, y_lim, 'TRIAL_dense_cs')
"""
n = 100000
m1 = 100
m2 = 10


"""
sparse_grid = np.array( [10, 10])
sparse_hc, sparse_cs = sparse_region(n, sparse_grid, m1)

normalize_colors(sparse_hc)
normalize_colors(sparse_cs)
x_lim = np.array([0.5, 1])
y_lim = np.array([0, 1])
heat_map_alt(sparse_hc, n, x_lim, y_lim, 'TRIAL_sparse_hc')
heat_map_alt(sparse_cs, n, x_lim, y_lim, 'TRIAL_sparse_cs')
"""


beta = 0.6
r = 0.6
params = [beta, r]
"""
hc, cs = find_thresholds(n, beta, r, m1, m2)
t_matrix = np.append([hc], [cs], 0)
#t_matrix = [[hc]
labels = [r'HC^+', r'$CsCsHM$']
histogram_comparison_save(t_matrix.transpose(), 'Thresholds', labels, params, n)
#"""

#"""
crit_hc, crit_cs = find_HCs(n, beta, r, m1, m2)

c_matrix = np.array([[crit_hc], [crit_cs]])

labels = [r'$H_0$ true ', r'$H_1$ true']
#histogram_comparison_save(c_matrix, 'comparison')
histogram_comparison_save(crit_hc, r'critical values for $HC^+$', labels, params, n)
histogram_comparison_save(crit_cs, r'critical values for $CsCsHM$', labels, params, n)
#"""
