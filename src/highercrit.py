# -*- coding: utf-8 -*-
# -*- coding: iso-8859-1 -*-
# -*- coding: latin-1 -*-
import numpy as np
import numpy as np
from scipy.stats import norm, chi2
from datageneration import *
from plotting import *


def hc_orthodox(x, beta, r, dist, plot=0):
    n = x.shape[0]
    pi = calculate_p_values(x, dist)
    sorted_pi, _ = sort_by_size(pi)
    hc_vector = np.zeros(n)
    n_float = float(n)

    for i in range(0, n):
        hc_vector[i] = np.sqrt(n) * \
                       (i/n_float - sorted_pi[i]) / np.sqrt(sorted_pi[i]*(1 - sorted_pi[i]))
    i_opt = np.argmax(hc_vector)
    hc_opt = np.max(hc_vector)
    # print('Optimal HC:', hc_opt, 'index i_opt:', i_opt)
    if plot == 1:
        save_figures(x, sorted_pi, hc_vector, hc_opt, i_opt, beta, r, 'orthodox')
    elif plot == 2:
        visualize_values(x, sorted_pi, hc_vector, hc_opt, i_opt)
    return i_opt, hc_opt


def hc_plus(x, beta, r, dist, plot=0):
    n = x.shape[0]
    pi = calculate_p_values(x, dist)
    sorted_pi, _ = sort_by_size(pi)
    hc_vector = np.zeros(n)
    n_float = float(n)

    for i in range(1, n):
        if sorted_pi[i] > 1/n:
            hc_vector[i] = np.sqrt(n_float) * \
                           (i/n_float - sorted_pi[i]) / np.sqrt(sorted_pi[i]*(1 - sorted_pi[i]))

    i_opt = np.argmax(hc_vector)
    hc_opt = np.max(hc_vector)
    # print('Optimal HC:', hc_opt, 'index i_opt:', i_opt)
    if plot == 1:
        save_figures(x, sorted_pi, hc_vector, hc_opt, i_opt, beta, r, 'plus')
    elif plot == 2:
        visualize_values(x, sorted_pi, hc_vector, hc_opt, i_opt)
    return i_opt, hc_opt


def hc_plus_classification(x, beta, r, dist, plot=0):
    n = x.shape[0]
    pi = calculate_two_sided_p_values(x, dist)
    sorted_pi, _ = sort_by_size(pi)
    hc_vector = np.zeros(n)
    n_float = float(n)

    for i in range(1, n):
        if sorted_pi[i] > 1/n:
            hc_vector[i] = np.sqrt(n_float) * \
                           (i/n_float - sorted_pi[i]) / np.sqrt(sorted_pi[i]*(1 - sorted_pi[i]))
    i_opt = np.argmax(hc_vector)
    hc_opt = np.max(hc_vector)
    # print('Optimal HC:', hc_opt, 'index i_opt:', i_opt)
    if plot == 1:
        save_figures(x, sorted_pi, hc_vector, hc_opt, i_opt, beta, r, 'plus')
    elif plot == 2:
        visualize_values(x, sorted_pi, hc_vector, hc_opt, i_opt)
    return i_opt, hc_opt


def hc_cscshm_1(x, beta, r, dist, plot=0):
    #  Csörgós Csörgós Horvath Mason
    n = x.shape[0]
    pi = calculate_p_values(x, dist)
    sorted_pi, _ = sort_by_size(pi)
    hc_vector = np.zeros(n)
    n_float = float(n)

    for i in range(1, n):
        if sorted_pi[i] > 1/n:
            pi_val = sorted_pi[i]
            norm = pi_val * (1 - pi_val)
            hc_vector[i] = np.sqrt(n_float) * (i/n_float - pi_val) / np.sqrt(norm * np.log(np.log(1 / norm)))
    i_opt = np.argmax(hc_vector)
    hc_opt = np.max(hc_vector)
    # print('Optimal HC:', hc_opt, 'index i_opt:', i_opt)
    if plot == 1:
        save_figures(x, sorted_pi, hc_vector, hc_opt, i_opt, beta, r, 'CsCsHM')
    elif plot == 2:
        visualize_values(x, sorted_pi, hc_vector, hc_opt, i_opt)
    return i_opt, hc_opt


def hc_cscshm_2(x, beta, r, dist, plot=0):
    #  Csörgós Csörgós Horvath Mason
    n = x.shape[0]
    pi = calculate_p_values(x, dist)
    sorted_pi, _ = sort_by_size(pi)
    hc_vector = np.zeros(n)
    n_float = float(n)

    for i in range(1, n):
        if sorted_pi[i] > 1/n:
            pi_val = sorted_pi[i]
            norm = pi_val * (1 - pi_val)
            hc_vector[i] = np.sqrt(n_float) * (i/n_float - pi_val) / (np.sqrt(norm) * np.log(np.log(1 / norm)))
    i_opt = np.argmax(hc_vector)
    hc_opt = np.max(hc_vector)
    # print('Optimal HC:', hc_opt, 'index i_opt:', i_opt)
    if plot == 1:
        save_figures(x, sorted_pi, hc_vector, hc_opt, i_opt, beta, r, 'CsCsHM')
    elif plot == 2:
        visualize_values(x, sorted_pi, hc_vector, hc_opt, i_opt)
    return i_opt, hc_opt


def hc_thresholding(x, y, beta, r, hc_function, threshold='hard', z_type=1):
    n, p = x.shape
    z = np.zeros(p)
    # Estimating the class means
    index_1 = y == -1
    index_2 = y == 1
    n_1 = np.sum(index_1)
    n_2 = np.sum(index_2)
    mu_1_est = 1 / n_1 * np.sum(x[index_1, ], 0)
    mu_2_est = 1 / n_2 * np.sum(x[index_2, ], 0)
    var_est = 1 / (n - 2) * (np.sum(np.power(x[index_1, ] - mu_1_est, 2), 0) + np.sum(np.power(x[index_2, ] - mu_2_est, 2), 0))
    std_est = np.sqrt(var_est)

    if z_type==0:
        # Z-scores from estimated feature means
        z = 1 / np.sqrt(1 / n_1 + 1 / n_2) * (mu_2_est - mu_1_est)
        z = np.divide(z, std_est)
        z = (z - np.mean(z)) / np.std(z)
    else:
        # Z-scores from def. Requires normalized data.
        for i in range(0, p):
            z[i] = np.sqrt(1 / n) * np.sum(np.multiply(x[:, i], y))


    if hc_function=='default':
        i_opt, _ = hc_plus_classification(z, beta, r, {1: 'norm'})
        # hc_opt = hc_vector[i_opt]
        z_opt = z[i_opt]
    else:
        i_opt, _ = hc_function(z, beta, r, {1: 'norm'})
        z_opt = z[i_opt]
    # print('Optimal HC:', hc_opt, 'index i_opt:', i_opt)
    # save_figures(z, sorted_pi, hc_vector, hc_opt, i_opt, beta, r, 'threshold')
    weights = np.zeros(p)
    if threshold == 'hard':
        for i in range(0, p):
            if z[i] > z_opt:
                weights[i] = z[i]
    elif threshold == 'soft':
        for i in range(0, p):
            if z[i] > z_opt:
                weights[i] = np.sign(z[i]) * np.max([np.abs(z[i]) - z_opt, 0])
    elif threshold == 'clip':
        for i in range(0, p):
            if z[i] > z_opt:
                weights[i] = np.sign(z[i]) * 1
    return z_opt, weights, mu_1_est, mu_2_est, std_est


def cscshm_thresholding(x, y, beta, r, hc=1, z_type=1):
    n, p = x.shape
    z = np.zeros(p)
    # Estimating the class means
    index_1 = y==0
    index_2 = y==1
    n_1 = np.sum(y==0)
    n_2 = np.sum(y==1)
    mu_1_est = 1/n_1 * np.sum(x[index_1, ], 0)
    mu_2_est = 1/n_2 * np.sum(x[index_2, ], 0)
    var_est = 1/(n-2) * (np.sum(np.power(x[index_1, ]-mu_1_est, 2), 0) + np.sum(np.power(x[index_2, ]-mu_2_est, 2), 0))
    std_est = np.sqrt(var_est)

    #Creating z-scores
    if z_type==0:
        # Z-scores from estimated feature means
        z = 1 / np.sqrt(1 / n_1 + 1 / n_2) * (mu_2_est - mu_1_est)
        z = np.divide(z, std_est)
        z = (z - np.mean(z)) / np.std(z)
    else:
        # Z-scores from def.
        z = np.zeros(p)
        for i in range(0, p):
            z[i] = np.sqrt(1 / n) * np.sum(np.multiply(x[:, i], y))

    # Finding the threshold
    pi = calculate_p_values(z, {1: 'norm'})
    sorted_pi, sorted_index = sort_by_size(pi)
    hc_vector = np.zeros(n)
    for i in range(1, n):
        if sorted_pi[i] > 1 / n:
            pi_val = sorted_pi[i]
            norm = pi_val * (1 - pi_val)
            if hc == 1:
                hc_vector[i] = np.sqrt(n) * (i / n - pi_val) / np.sqrt(norm * np.log(np.log(1 / norm)))
            else:
                hc_vector[i] = np.sqrt(n) * (i / n - pi_val) / (np.sqrt(norm) * np.log(np.log(1 / norm)))

    i_opt = np.argmax(hc_vector)
    #hc_opt = np.max(hc_vector)
    s_i_opt = np.extract(sorted_index==i_opt, sorted_index)[0]

    # Creating the selection vector
    t_crit = z[s_i_opt]
    selection_vec = (z > t_crit)

    selection_vec = selection_vec.astype(int)
    return selection_vec, mu_1_est, mu_2_est, std_est


def cscshm_discriminant_rule(x, selection_vec, mu_1_est, mu_2_est, std_est):
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


# mu_est = (mu_1_est + mu_2_est) / 2
def discriminant_rule(weights, x, mu1, mu2, std_est):
    n, p = x.shape
    y = np.zeros(n)
    mu_est = (mu1 + mu2)/2
    mu_tiled = np.tile(mu_est, (n, 1))
    std_tiled = np.tile(std_est, (n, 1))
    x = (x - mu_tiled) / std_tiled

    # Exchange for matrix version
    for i in range(0, n):
        z = np.sum(weights*x[i, ])
        if z > 0:
            y[i] = 1
        else:
            y[i] = -1
    return y


def calculate_p_values(x, dist):  # one sided normal/chi2 p-values
    n = x.shape[0]
    p_values = np.zeros(n)
    if dist[1] == 'norm':
        p_values = norm.sf(x)
    elif dist[1] == 'chi2':
        degrees_freedom = dist['df']
        location = dist['loc']
        p_values = chi2.sf(x, degrees_freedom, location)
    return p_values


def calculate_two_sided_p_values(x, dist):  # one sided normal/chi2 p-values
    n = x.shape[0]
    p_values = np.zeros(n)
    if dist[1] == 'norm':
        p_values = 2*norm.cdf(-abs(x))
    elif dist[1] == 'chi2':
        print('not implemented yet..')
        degrees_freedom = dist['df']
        location = dist['loc']
        p_values = chi2.sf(x, degrees_freedom, location)
    return p_values


def sort_by_size(pi):
    # For i in range(0, pi.shape[0]):
    sort_index = np.argsort(pi)
    pi = np.sort(pi)
    return pi, sort_index
