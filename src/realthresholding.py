# -*- coding: utf-8 -*-
# -*- coding: iso-8859-1 -*-
# -*- coding: latin-1 -*-
import numpy as np
from scipy.stats import norm

def hc_plus_classification(pi, alpha):
    n = pi.shape[0]
    hc_vector = np.zeros(n)
    n_float = float(n)
    trunc = np.floor(alpha * n).astype(int)
    for i in range(0, trunc):
        if pi[i] > 1/n:
            #hc_vector[i] = np.sqrt(n_float) * (i/n_float - pi[i]) / np.sqrt(pi[i]*(1 - pi[i]))
            hc_vector[i] = np.sqrt(n_float) * ((i+1)/n_float - pi[i]) / np.sqrt((i+1)/n_float*(1 - (i+1)/n_float))

    i_opt = np.argmax(hc_vector)
    pi_opt = pi[i_opt]
    hc_opt = np.max(hc_vector)
    return pi_opt, hc_opt, i_opt


def hc_thresholding(x, y, threshold='clip', alpha=0.5, stdflag=0, muflag=0):
    n, p = x.shape
    z = np.zeros(p)
    # Estimating the class means
    index_1 = (y == -1)
    index_2 = (y == 1)
    n_1 = np.sum(index_1)
    n_2 = np.sum(index_2)
    mu_1_est = (1 / n_1) * np.sum(x[index_1, ], 0)
    mu_2_est = (1 / n_2) * np.sum(x[index_2, ], 0)
    var_est = (1 / (n - 2)) * (np.sum(np.power(x[index_1, ] - mu_1_est, 2), 0) + np.sum(np.power(x[index_2, ] - mu_2_est, 2), 0))
    std_est = np.sqrt(var_est)

    med_std = np.median(std_est)
    if stdflag==0:
        std_est = std_est + med_std
    elif stdflag==1:
        std_est = np.maximum(std_est, med_std)

    # Z-scores from estimated feature means
    z = (1 / np.sqrt(1/n_1 + 1/n_2) ) * (mu_2_est - mu_1_est)
    z = np.divide(z, std_est)
    z = (z - np.mean(z)) / np.std(z)

    pi = calculate_two_sided_p_values(z)
    sorted_pi, sorted_index = sort_by_size(pi)

    #""""
    pi_opt, hc_opt, max_hc_ind = hc_plus_classification(sorted_pi, alpha)
    opt_index = np.where(pi == pi_opt)
    z_opt = z[opt_index]
    print('For-loop: pi_opt', pi_opt, 'max_hc_ind', max_hc_ind, 'hc_opt' , hc_opt, 'opt_index', opt_index, 'zopt', z_opt)

    #"""
    #"""# Vectorized version of the HC-objective function
    ind = np.where(sorted_pi <= 1/p)
    ii = (1+np.arange(p)) / (p+1)
    hc_vector = np.sqrt(p) * (ii - sorted_pi) / np.sqrt(ii - np.power(ii, 2))
    hc_vector[ind] = 0
    hc_vector = hc_vector[0:np.floor(alpha*p).astype(int)]

    max_hc_ind = np.argmax(hc_vector)
    opt_z_index = np.where(pi == sorted_pi[max_hc_ind])
    z_opt = z[opt_z_index]
    pi_opt = sorted_pi[max_hc_ind]
    hc_opt = hc_vector[max_hc_ind]
    opt_index = opt_z_index
    print('Vectorized: pi_opt', pi_opt, 'max_hc_ind', max_hc_ind, 'hc_opt' , hc_opt, 'opt_index', opt_index, 'zopt', z_opt)
    #"""


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
            if abs(z[i]) > abs(z_opt):
                weights[i] = np.sign(z[i])
    return z_opt, weights, mu_1_est, mu_2_est, std_est


# mu_est = (mu_1_est + mu_2_est) / 2
def discriminant_rule(weights, x, mu1, mu2, std_est):
    n, p = x.shape
    y = np.zeros(n)
    mu_est = (mu1 + mu2)/2
    mu_tiled = np.tile(mu_est, (n, 1))
    std_tiled = np.tile(std_est, (n, 1))
    x = (x - mu_tiled) / std_tiled

    # Exchange for matrix version!!!!
    for i in range(0, n):
        z = np.sum(weights*x[i, ])
        if z > 0:
            y[i] = 1
        else:
            y[i] = -1
    return y


def hc_cscshm_1(x):
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

    return i_opt, hc_opt


def hc_cscshm_2(x):
    #  Csörgós Csörgós Horvath Mason
    n = x.shape[0]
    pi = calculate_p_values(x)
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

    return i_opt, hc_opt


def cscshm_thresholding(x, y, hc=1, alpha=0.5, stdflag=0):
    n, p = x.shape
    # Estimating the class means
    n_1 = np.sum(y == -1)
    if n_1 == 0:
        n_1 = np.sum(y == 0)
        index_1 = y == 0
    else:
        index_1 = y == -1

    n_2 = np.sum(y == 1)
    index_2 = y==1

    mu_1_est = (1 / n_1) * np.sum(x[index_1,], 0)
    mu_2_est = (1 / n_2) * np.sum(x[index_2,], 0)
    var_est = (1 / (n - 2)) * (np.sum(np.power(x[index_1,] - mu_1_est, 2), 0) + np.sum(np.power(x[index_2,] - mu_2_est, 2), 0))
    std_est = np.sqrt(var_est)

    med_std = np.median(std_est)
    if stdflag == 0:
        std_est = std_est + med_std
    elif stdflag == 1:
        std_est = np.maximum(std_est, med_std)

    #Creating z-scores
    z = 1 / np.sqrt(1 / n_1 + 1 / n_2) * (mu_2_est - mu_1_est)
    z = np.divide(z, std_est)
    z = (z - np.mean(z)) / np.std(z)

    # Finding the threshold
    pi = calculate_p_values(z)
    sorted_pi, sorted_index = sort_by_size(pi)
    hc_vector = np.zeros(p)

    trunc = np.floor(alpha * p).astype(int)
    for i in range(0, trunc):
        if sorted_pi[i] > 1 / p:
            pi_val = sorted_pi[i]
            norm = pi_val * (1 - pi_val)
            if hc == 1:
                hc_vector[i] = np.sqrt(p) * ((i+1)/p - pi_val) / np.sqrt(norm * np.log(np.log(1 / norm)))
            else:
                hc_vector[i] = np.sqrt(p) * ((i+1)/p - pi_val) / (np.sqrt(norm) * np.log(np.log(1 / norm)))


    max_hc_ind = np.argmax(hc_vector)
    hc_opt = hc_vector[max_hc_ind]
    pi_opt = sorted_pi[max_hc_ind]
    opt_index = np.where(pi == pi_opt)
    z_opt = z[opt_index]
    print('For-loop: pi_opt', pi_opt, 'max_hc_ind', max_hc_ind,'hc_opt', hc_opt, 'opt_index', opt_index, 'zopt', z_opt)

    # Vectorized version of finding HC
    # To be verified
    #"""
    ii = (1+np.arange(p, dtype=float)) / (p+1)
    norm_alt = sorted_pi - np.power(sorted_pi, 2)
    if hc==1:
        hc_alt = np.sqrt(p) * np.divide((ii - sorted_pi), np.sqrt(norm_alt * np.log(np.log(1/norm_alt))))
    else:
        hc_alt = np.sqrt(p) * np.divide((ii - sorted_pi), np.sqrt(norm_alt) * np.log(np.log(1/norm_alt)))

    ind = np.where(sorted_pi <= 1/p)
    hc_alt[ind] = 0
    hc_alt = hc_alt[0:np.floor(alpha * p).astype(int)]

    max_hc_ind = np.argmax(hc_alt)
    opt_z_index = np.where(pi == sorted_pi[max_hc_ind])
    z_opt = z[opt_z_index]
    pi_opt = sorted_pi[max_hc_ind]
    hc_opt = np.max(hc_alt)
    print('Vectorized: pi_opt', pi_opt, 'max_hc_ind', max_hc_ind, 'hc_opt', hc_opt, 'opt_index', opt_z_index, 'zopt', z_opt)

    #"""


    selection_vec = (abs(z) > z_opt)

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


def calculate_p_values(x):  # one sided normal/chi2 p-values
    n = x.shape[0]
    p_values = norm.sf(x)
    return p_values


def calculate_two_sided_p_values(x):  # one sided normal/chi2 p-values
    p_values = 2*(1 - norm.cdf(abs(x)))
    return p_values


def sort_by_size(pi):
    sort_index = np.argsort(pi)
    sorted_pi = np.sort(pi)
    return sorted_pi, sort_index
