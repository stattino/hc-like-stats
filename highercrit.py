import numpy as np
from scipy.stats import norm, chi2
from datageneration import *
from plotting import *


def hc_orthodox(x, beta, r, dist, plot=0):
    n = x.shape[0]
    pi = calculate_p_values(x, dist)
    sorted_pi = sort_by_size(pi)
    hc_vector = np.zeros(n)
    for i in range(0, n):
        hc_vector[i] = np.sqrt(n) * \
                       (i/n - sorted_pi[i]) / np.sqrt(sorted_pi[i]*(1 - sorted_pi[i]))
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
    sorted_pi = sort_by_size(pi)
    hc_vector = np.zeros(n)
    for i in range(1, n):
        if sorted_pi[i] > 1/n:
            hc_vector[i] = np.sqrt(n) * \
                           (i/n - sorted_pi[i]) / np.sqrt(sorted_pi[i]*(1 - sorted_pi[i]))
    i_opt = np.argmax(hc_vector)
    hc_opt = np.max(hc_vector)
    # print('Optimal HC:', hc_opt, 'index i_opt:', i_opt)
    if plot == 1:
        save_figures(x, sorted_pi, hc_vector, hc_opt, i_opt, beta, r, 'plus')
    elif plot == 2:
        visualize_values(x, sorted_pi, hc_vector, hc_opt, i_opt)
    return i_opt, hc_opt


def hc_cscshm(x, beta, r, dist, plot=0):
    #  Csörgós Csörgós Horvath Mason
    n = x.shape[0]
    pi = calculate_p_values(x, dist)
    sorted_pi = sort_by_size(pi)
    hc_vector = np.zeros(n)
    for i in range(1, n):
        if sorted_pi[i] > 1/n:
            pi_val = sorted_pi[i]
            norm = pi_val * (1 - pi_val)
            hc_vector[i] = np.sqrt(n) * (i/n - pi_val) / np.sqrt(norm * np.log(np.log(1 / norm)))
    i_opt = np.argmax(hc_vector)
    hc_opt = np.max(hc_vector)
    # print('Optimal HC:', hc_opt, 'index i_opt:', i_opt)
    if plot == 1:
        save_figures(x, sorted_pi, hc_vector, hc_opt, i_opt, beta, r, 'CsCsHM')
    elif plot == 2:
        visualize_values(x, sorted_pi, hc_vector, hc_opt, i_opt)
    return i_opt, hc_opt


def hc_thresholding(x, y, beta, r, threshold='hard'):
    n, p = x.shape
    z = np.zeros(p)
    for i in range(0, p):
        z[i] = np.sqrt(1/n)*np.sum(np.multiply(x[:, i], y))
    pi = calculate_p_values(z, {1: 'norm'})
    sorted_pi = sort_by_size(pi)
    hc_vector = np.zeros(p)
    print('Data generated')

    for i in range(1, p):
        hc_vector[i] = np.sqrt(p) * (i/p - sorted_pi[i]) / np.sqrt(i/p * (1 - i/p))

    print('HC statistic calculated')
    i_opt = np.argmax(hc_vector)
    hc_opt = hc_vector[i_opt]
    z_opt = z[i_opt]
    print('Optimal HC:', hc_opt, 'index i_opt:', i_opt)

    save_figures(z, sorted_pi, hc_vector, hc_opt, i_opt, beta, r, 'threshold')

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

    return z_opt, weights


def discriminant_rule(weights, x):
    n, p = x.shape
    y = np.zeros(n)
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
        #for i in range(0, n):
         #   p_values[i] = norm.sf(x[i])
    elif dist[1] == 'chi2':
        degrees_freedom = dist['df']
        location = dist['loc']
        p_values = chi2.sf(x, degrees_freedom, location)
    return p_values


def sort_by_size(pi):
    # For i in range(0, pi.shape[0]):
    pi = np.sort(pi)
    return pi
