# -*- coding: utf-8 -*-
# -*- coding: iso-8859-1 -*-
# -*- coding: latin-1 -*-
import numpy as np
from scipy.stats import norm


def hc_thresholding(x, y, alpha=0.5):
    n, p = x.shape
    z = np.zeros(p)
    # Estimating the class means
    n_float = float(n)
    p_float = float(p)
    for i in range(0, p):
        z[i] = np.sqrt(1 / n_float) * np.sum(np.multiply(x[:, i], y))

    pi = calculate_two_sided_p_values(z)
    sorted_pi, _ = sort_by_size(pi)

    ii = (1 + np.arange(p)) / (p_float + 1)
    hc_vector = np.sqrt(p_float) * (ii - sorted_pi) / np.sqrt(ii - np.power(ii, 2))

    ind = np.where(sorted_pi <= 1 / p_float)
    hc_vector[ind] = 0

    hc_vector = hc_vector[0:np.floor(alpha * p).astype(int)]

    max_hc_ind = np.argmax(hc_vector)
    opt_z_index = np.where(pi == sorted_pi[max_hc_ind])
    z_opt = z[opt_z_index]
    #pi_opt = sorted_pi[max_hc_ind]
    #hc_opt = hc_vector[max_hc_ind]
    #opt_index = opt_z_index

    weights = np.zeros(p)
    """if threshold == 'hard':
        for i in range(0, p):
            if z[i] > z_opt:
                weights[i] = z[i]
    elif threshold == 'soft':
        for i in range(0, p):
            if z[i] > z_opt:
                weights[i] = np.sign(z[i]) * np.max([np.abs(z[i]) - z_opt, 0])
    elif threshold == 'clip':"""
    for i in range(0, p):
        if np.abs(z[i]) >= np.abs(z_opt):
            weights[i] = np.sign(z[i])
    return weights   #, mu_1_est, mu_2_est, std_est



def cscshm_thresholding(x, y, hc=1, alpha=0.5):
    n, p = x.shape

    n_float = float(n)
    p_float = float(p)

    z = np.zeros(p)
    for i in range(0, p):
        z[i] = np.sqrt(1 / n_float) * np.sum(np.multiply(x[:, i], y))

    # Finding the threshold
    pi = calculate_two_sided_p_values(z) # under test
    sorted_pi, sorted_index = sort_by_size(pi)

    ii = (1 + np.arange(p, dtype=float)) / (p_float + 1)
    norm_alt = sorted_pi - np.power(sorted_pi, 2)
    if hc == 1:
        hc_alt = np.sqrt(p_float) * np.divide((ii - sorted_pi), np.sqrt(norm_alt * np.log(np.log(1 / norm_alt))))
    else:
        hc_alt = np.sqrt(p_float) * np.divide((ii - sorted_pi), np.sqrt(norm_alt) * np.log(np.log(1 / norm_alt)))

    # How small can we cut?
    #ind = np.where(sorted_pi <= 1 / np.power(p_float, 2))
    #hc_alt[ind] = 0

    hc_alt = hc_alt[0:np.floor(alpha * p_float).astype(int)]

    max_hc_ind = np.argmax(hc_alt)
    opt_z_index = np.where(pi == sorted_pi[max_hc_ind])
    z_opt = z[opt_z_index]
    pi_opt = sorted_pi[max_hc_ind]
    hc_opt = np.max(hc_alt)

    selection_vec = (np.abs(z) >= np.abs(z_opt))
    #z_sort, _ = sort_by_size(z)
    #z_sort = abs(z_sort[::-1])
    selection_vec = selection_vec.astype(int)
    return selection_vec


def cscshm_thresholding_2(x, y, hc=1, alpha=0.5):
    n, p = x.shape

    n_float = float(n)
    p_float = float(p)

    n_1 = np.sum(y == -1)
    if n_1 == 0:
        n_1 = np.sum(y == 0)
        index_1 = y == 0
    else:
        index_1 = y == -1

    n_2 = np.sum(y == 1)
    index_2 = y == 1

    mu_1_est = (1 / n_1) * np.sum(x[index_1,], 0)
    mu_2_est = (1 / n_2) * np.sum(x[index_2,], 0)
    var_est = (1 / (n - 2)) * (np.sum(np.power(x[index_1,] - mu_1_est, 2), 0) + np.sum(np.power(x[index_2,] - mu_2_est, 2), 0))
    std_est = np.sqrt(var_est)

    z = np.zeros(p)
    for i in range(0, p):
        z[i] = np.sqrt(1 / n_float) * np.sum(np.multiply(x[:, i], y))

    # Finding the threshold
    pi = calculate_p_values(z) # under test
    sorted_pi, sorted_index = sort_by_size(pi)

    ii = (1 + np.arange(p, dtype=float)) / (p_float + 1)
    norm_alt = sorted_pi - np.power(sorted_pi, 2)
    if hc == 1:
        hc_alt = np.sqrt(p_float) * np.divide((ii - sorted_pi), np.sqrt(norm_alt * np.log(np.log(1 / norm_alt))))
    else:
        hc_alt = np.sqrt(p_float) * np.divide((ii - sorted_pi), np.sqrt(norm_alt) * np.log(np.log(1 / norm_alt)))

    # How small can we cut?
    #ind = np.where(sorted_pi <= 1 / np.power(p_float, 2))
    #hc_alt[ind] = 0

    hc_alt = hc_alt[0:np.floor(alpha * p_float).astype(int)]

    max_hc_ind = np.argmax(hc_alt)
    opt_z_index = np.where(pi == sorted_pi[max_hc_ind])
    z_opt = z[opt_z_index]
    pi_opt = sorted_pi[max_hc_ind]
    hc_opt = np.max(hc_alt)

    selection_vec = (abs(z) >= (z_opt))
    #z_sort, _ = sort_by_size(z)
    #z_sort = abs(z_sort[::-1])
    selection_vec = selection_vec.astype(int)
    return selection_vec, mu_1_est, mu_2_est, std_est


def classification_region(theta, p, grid, m1, cs=1, alpha=0.5, t_type='hc'):
    dense = np.zeros(grid)
    grid_x = np.linspace(0, 1-theta, grid[0])
    grid_y = np.linspace(0, 1, grid[1])
    for beta in range(0, grid[0]):
        for r in range(0, grid[1]):
            #print('beta', grid_x[beta], 'r', grid_y[r])
            #if (r > 0.2) and (beta < 0.2):
            #    dense[beta, r] = 0
            #else:
            dense[beta, r] = classification_error(theta, grid_x[beta], grid_y[r], p, m1, cs, alpha, t_type)
            print('Fraction of region completed', r + beta*grid[1] + 1, '/', grid[0]*grid[1])
    return dense


def classification_error(theta, beta, r, p, m1, cs=1, alpha=0.5, t_type='hc'):
    error = 0.
    for i in range(0, m1):
        if t_type=='cs':
            x_train, y_train, x_test, y_test = generate_classification_data_cscshm(p, theta, beta, r)
            selected = cscshm_thresholding(x_train, y_train, cs, alpha)
            y_attempt = cscshm_discriminant_rule(x_test, selected, r)

            #selected, mu_1_est, mu_2_est, std_est = cscshm_thresholding_2(x_train, y_train, cs, alpha)
            #y_attempt = cscshm_discriminant_rule_2(x_test, selected,  mu_1_est, mu_2_est, std_est)

        elif t_type=='hc':
            x_train, y_train, x_test, y_test = generate_classification_data(p, theta, beta, r)
            weights = hc_thresholding(x_train, y_train, alpha)
            y_attempt = discriminant_rule(x_test, weights)

        #print(np.sum(weights!=1), np.sum(weights==1))
        test_size = float(y_test.shape[0])
        part_error = sum(y_attempt != y_test) / test_size
        error += part_error
    m1 = float(m1)
    #print(error)
    #print(m1)
    return error/m1


def cscshm_discriminant_rule(x, selection_vec, r):
    n, p = x.shape
    tau = np.sqrt(2 * r * np.log(p))
    mu0 = 2*tau/np.sqrt(n)   # *2 ???
    x_shift = (x - 0.5*mu0)*mu0
    w_0 = np.dot(x_shift, selection_vec.transpose())
    w_0 = w_0 > 0
    w_0 = w_0.astype(int)
    return w_0


def cscshm_discriminant_rule_2(x, selection_vec, mu_1_est, mu_2_est, std_est):
    n, p = x.shape
    mu_est = (mu_1_est + mu_2_est) /2
    mu_tiled = np.tile(mu_est, (n, 1))
    std_tiled = np.tile(std_est, (n, 1))
    #x = (x - mu_tiled)/std_tiled

    x_shift = np.subtract(x, 1/2*(mu_1_est + mu_2_est))*(mu_1_est - mu_2_est)
    w_0 = np.dot(x_shift, selection_vec.transpose())
    w_0 = w_0 < 0
    w_0 = w_0.astype(int)

    return w_0
def discriminant_rule(x, weights):
    y = np.dot(x, weights.transpose())
    y = y > 0
    y = y.astype(int)
    ind = np.where(y == 0)
    y[ind] = -1
    return y


def calculate_p_values(x):  # one sided normal/chi2 p-values
    p_values = norm.sf(x)
    return p_values


def calculate_two_sided_p_values(x):  # one sided normal/chi2 p-values
    p_values = 2*(1 - norm.cdf(abs(x)))
    return p_values


def sort_by_size(pi):
    sort_index = np.argsort(pi)
    pi = np.sort(pi)
    return pi, sort_index


def generate_classification_data_cscshm(p, theta, beta, r, balance=0.5):
    n = int(np.ceil(2 * np.power(p, theta)))
    n_float = float(n)
    tau = np.sqrt(2 * r * np.log(p))
    epsilon = np.power(p, -beta)
    mu0 = 2*tau/np.sqrt(n_float)

    n_signals = int(np.ceil(epsilon * p))
    index = np.random.choice(p, n_signals, False)

    x_train = np.zeros([n, p])
    y_train = np.zeros(n)
    x_test = np.zeros([n, p])
    y_test = np.zeros(n)

    n_class_2 = int(np.ceil(balance*n))
    index_class_2 = np.random.choice(n, n_class_2, False)

    y_train[index_class_2] = 1
    y_test[index_class_2] = 1

    for i in range(0, n):
        x_train[i, ] = np.random.normal(0, 1, p)
        x_train[i, index] = np.random.normal(mu0 * y_train[i], 1, n_signals)
        x_test[i,] = np.random.normal(0, 1, p)
        x_test[i, index] = np.random.normal(mu0 * y_test[i], 1, n_signals)

    return x_train, y_train, x_test, y_test


def generate_classification_data(p, theta, beta, r, balance=0.5):
    n = int(np.ceil(2 * np.power(p, theta)))
    n_float = float(n)

    tau = np.sqrt(2 * r * np.log(p))
    epsilon = np.power(p, -beta)
    mu0 = tau/np.sqrt(n_float)

    n_signals = int(np.ceil(epsilon * p))
    index = np.random.choice(p, n_signals, False)

    x_train = np.zeros([n, p])
    y_train = np.ones(n)
    x_test = np.zeros([n, p])
    y_test = np.ones(n)

    n_class_2 = int(np.ceil(balance*n))
    index_class_2 = np.random.choice(n, n_class_2, False)

    y_train[index_class_2] *= -1
    y_test[index_class_2] *= -1

    for i in range(0, n):
        x_train[i, ] = np.random.normal(0, 1, p)
        x_train[i, index] = np.random.normal(mu0 * y_train[i], 1, n_signals)
        x_test[i,] = np.random.normal(0, 1, p)
        x_test[i, index] = np.random.normal(mu0 * y_test[i], 1, n_signals)

    return x_train, y_train, x_test, y_test
