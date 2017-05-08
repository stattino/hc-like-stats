import numpy as np
from scipy.stats import norm, chi2
from time import gmtime, strftime


def generate_data(n, beta, r, signal_presence, dist):
    if dist[1] == 'norm':
        # dist['var'] for later implementation?
        return generate_normal_mixture(n, beta, r, signal_presence)
    elif dist[1] == 'chi2':
        return generate_chi2_mixture(n, beta, r, signal_presence, dist['df'], dist['loc'])
    else:
        print('check distribution')


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


def generate_chi2_mixture(n, beta, r, signal_presence, df=1, location=0):
    epsilon = np.power(n, -beta)
    if beta > 0.5:
        w0 = 2 * r * np.log(n)
    else:
        w0 = np.power(n, -2*r)
    x = chi2.rvs(df, location, 1, n)
    if signal_presence == 1:
        n_signals = int(np.ceil(epsilon*n))
        n = int(n)
        index = np.random.choice(n, n_signals, False)
        x[index] = chi2.rvs(df, location+w0, 1, n_signals)
    return x


def generate_classification_data(p, theta, beta, r, balance=0.5):
    n = int(np.ceil(2 * np.power(p, theta)))
    if p < 2*n:
        print('...not considering p>>n...')
    tau = np.sqrt(2 * r * np.log(p))
    epsilon = np.power(p, -beta)
    mu0 = tau/np.sqrt(n)

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
        x_train[i, index] = np.random.normal(mu0*y_train[i], 1, n_signals)
        x_test[i,] = np.random.normal(0, 1, p)
        x_test[i, index] = np.random.normal(mu0 * y_test[i], 1, n_signals)

    return x_train, y_train, x_test, y_test



def heat_map_save(matrix, n, m, msg):
    time = strftime("%m-%d_%H-%M-%S", gmtime())
    filename = 'data/heatmap_data_{}_grid={}x{}_time_{}.txt'.format(msg, n, m, time)
    print(filename)
    np.savetxt(filename, matrix)
    return


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


def hc_cscshm_1(x, beta, r, dist, plot=0):
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


def hc_cscshm_2(x, beta, r, dist, plot=0):
    #  Csörgós Csörgós Horvath Mason
    n = x.shape[0]
    pi = calculate_p_values(x, dist)
    sorted_pi = sort_by_size(pi)
    hc_vector = np.zeros(n)
    for i in range(1, n):
        if sorted_pi[i] > 1/n:
            pi_val = sorted_pi[i]
            norm = pi_val * (1 - pi_val)
            hc_vector[i] = np.sqrt(n) * (i/n - pi_val) / (np.sqrt(norm) * np.log(np.log(1 / norm)))
    i_opt = np.argmax(hc_vector)
    hc_opt = np.max(hc_vector)
    # print('Optimal HC:', hc_opt, 'index i_opt:', i_opt)
    if plot == 1:
        save_figures(x, sorted_pi, hc_vector, hc_opt, i_opt, beta, r, 'CsCsHM')
    elif plot == 2:
        visualize_values(x, sorted_pi, hc_vector, hc_opt, i_opt)
    return i_opt, hc_opt


def hc_thresholding(x, y, beta, r, hc_function, threshold='hard'):
    n, p = x.shape
    z = np.zeros(p)
    for i in range(0, p):
        z[i] = np.sqrt(1/n)*np.sum(np.multiply(x[:, i], y))

    if hc_function=='default':
        pi = calculate_p_values(z, {1: 'norm'})
        sorted_pi = sort_by_size(pi)
        hc_vector = np.zeros(p)
        # print('Data generated')

        for i in range(1, p):
            hc_vector[i] = np.sqrt(p) * (i/p - sorted_pi[i]) / np.sqrt(i/p * (1 - i/p))

        # print('HC statistic calculated')
        i_opt = np.argmax(hc_vector)
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


def calculate_two_sided_p_values(x, dist):  # one sided normal/chi2 p-values
    n = x.shape[0]
    p_values = np.zeros(n)
    if dist[1] == 'norm':
        p_values = 2*norm.cdf(-x)
        #for i in range(0, n):
         #   p_values[i] = norm.sf(x[i])
    elif dist[1] == 'chi2':
        print('not implemented yet..')
        degrees_freedom = dist['df']
        location = dist['loc']
        p_values = chi2.sf(x, degrees_freedom, location)
    return p_values


def sort_by_size(pi):
    # For i in range(0, pi.shape[0]):
    pi = np.sort(pi)
    return pi


def classification_detection_boundary(theta, p, grid, m1, m2, hc_function):
    error_matrix = classification_region(theta, p, grid, m1, m2, hc_function)
    x_lim = np.array([0, 1-theta])
    y_lim = np.array([0, 1])
    heat_map_save(error_matrix, grid[0], grid[1], 'classification')
    normalize_colors(error_matrix)
    heat_map_alt(error_matrix, p, x_lim, y_lim, 'classification')


def classification_region(theta, p, grid, m1, m2, hc_function):
    dense = np.zeros(grid)
    grid_x = np.linspace(0, 1-theta, grid[0])
    grid_y = np.linspace(0, 1, grid[1])
    for beta in range(0, grid[0]):
        for r in range(0, grid[1]):
            # dense[beta, r] = error_rate(grid_x[beta], grid_y[r], m1, m2, dist)
            if (r > 0.2 ) and (beta < 0.35):
                dense[beta, r] = 0
            else:
                dense[beta, r] = classification_error(theta, grid_x[beta], grid_y[r], p, m1, m2, hc_function)
            print('Fraction of region completed', r + beta*grid[1] + 1, '/', grid[0]*grid[1])
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
    heat_map_save(dense, grids[0], grids[1], 'dense_matrix')
    normalize_colors(dense)
    heat_map_alt(dense, m1, x_lim, y_lim, 'dense')

    sparse = sparse_region(n, grids[2:4], m1, m2, dist, hc_function)
    print('sparse region complete')
    x_lim = np.array([0.5, 1])
    y_lim = np.array([0, 1])
    heat_map_save(sparse, grids[2], grids[3], 'sparse_matrix')
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
            print('Fraction of dense region completed', r + beta*grid[1] + 1, '/', grid[0]*grid[1])
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
