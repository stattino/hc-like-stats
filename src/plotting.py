import numpy as np
import matplotlib.pyplot as plt
from time import gmtime, strftime
from scipy.stats import norm

def save_figures(z, pi, hcs, hc_opt, i_opt, beta, r, alpha, msg):

    display_percentage = 0.1
    time = strftime("%m-%d_%H-%M-%S", gmtime())
    n = pi.shape[0]
    new_n = int(np.floor(display_percentage*n))

    x_points = np.linspace(0, display_percentage, new_n, True)
    """if n<1000000:
        x_points = np.arange(0, alpha, 1 / n)
    else:
        x_points = np.arange(0, alpha*n, 1)
    """
    z = np.sort(z)
    z = z[::-1]

    fig = plt.figure()
    #print(x_points.shape, z.shape)

    plt.subplot(3, 1, 1)  # Z-scores
    plt.plot(x_points, z[0:new_n])
    plt.axvline(x=i_opt/n, ymax=z[i_opt]/np.max(z), color='#d62728')
    plt.axhline(y=z[i_opt], xmin=0, xmax=i_opt / (n*display_percentage), color='#d62728')
    plt.axis([0, display_percentage, 0, np.max(z)])
    plt.xticks(np.linspace(0, display_percentage, 6))
    plt.title('ordered Z-scores')

    plt.subplot(3, 1, 2)  # P-values
    plt.plot(x_points, pi[0:new_n])
    plt.axvline(x=i_opt / n, ymax=pi[i_opt]/np.max(pi), color='#d62728')
    plt.axis([0, display_percentage, 0, np.max(pi[0:new_n])])
    plt.xticks(np.linspace(0, display_percentage, 6))
    plt.title('ordered P-values')


    plt.subplot(3, 1, 3)  # HC-statistic
    plt.plot(x_points, hcs[0:new_n])
    plt.axvline(x=i_opt / n, ymax=(hc_opt + np.abs(np.min(hcs)))/(1.1*hc_opt - np.min(hcs)), color='#d62728')
    #plt.axhline(y=hc_opt, xmin=0, xmax=i_opt / (n*alpha), color='#d62728')
    plt.axis([0, display_percentage, np.min(hcs), 1.1*hc_opt])
    plt.xticks(np.linspace(0, display_percentage, 6))
    plt.title('Objective function')

    plt.tight_layout()

    filename = '../plots/hctest/{}-N={}-beta={}-r={}_alpha={}_time{}.png'.format(msg, n, beta, r, alpha, time)
    print(filename)
    fig.savefig(filename)
    return


def visualize_values(z, pi, hcs, hc_opt, i_opt):
    n = pi.shape[0]
    x_points = np.arange(0, 1, 1/n)
    z = np.sort(z)

    plt.subplot(3, 1, 1)
    plt.plot(x_points, z[::-1])

    plt.subplot(3, 1, 2)
    plt.plot(x_points, pi)

    plt.subplot(3, 1, 3)
    plt.plot(x_points, hcs)

    plt.axvline(x=i_opt/n, ymax=hc_opt, color='#d62728')
    plt.axhline(y=hc_opt, xmin=0, xmax=i_opt/n, color='#d62728')
    plt.show()
    return


def heat_map(dense, sparse, n):
    fig = plt.figure()
    nx, ny = dense.shape
    x = np.linspace(0, 0.5, nx)
    y = np.linspace(0, 0.5, ny)
    xv, yv = np.meshgrid(x, y)
    plt.pcolormesh(xv, yv, dense.transpose(), cmap='coolwarm')

    nx, ny = sparse.shape
    x = np.linspace(0.5, 1, nx)
    y = np.linspace(0, 1, ny)
    xv, yv = np.meshgrid(x, y)
    plt.pcolormesh(xv, yv, sparse.transpose(), cmap='coolwarm')  #seismic, RdYlBu
    plt.colorbar()

    time = strftime("%m-%d_%H-%M-%S", gmtime())
    filename = 'plots/heatmaps/Detection_Boundary_n={}_time_{}.png'.format(n, time)
    print(filename)
    fig.savefig(filename)
    return


def heat_map_alt(matrix, n, x_lim, y_lim, msg):
    fig = plt.figure()
    nx, ny = matrix.shape
    x = np.linspace(x_lim[0], x_lim[1], nx+1)
    y = np.linspace(y_lim[0], y_lim[1], ny+1)
    xv, yv = np.meshgrid(x, y)
    plt.pcolormesh(xv, yv, matrix.transpose(), cmap='RdYlBu_r')
    plt.xlim(x_lim[0], x_lim[1])
    plt.ylim(y_lim[0], y_lim[1])
    plt.colorbar()
    time = strftime("%m-%d_%H-%M-%S", gmtime())
    filename = '../plots/heatmaps/Detection_Boundary_n={}_grid={}x{}_time_{}_{}.png'.format(n, nx, ny, time, msg)
    print(filename)
    fig.savefig(filename)
    return


def histogram_save(vector, msg):
    n = vector.shape[0]
    fig, ax = plt.subplots()
    ax.hist(vector,bins=n)
    time = strftime("%m-%d_%H-%M-%S", gmtime())
    filename = '../plots/heatmaps/Histogram_n={}_time_{}_{}.png'.format(n, time, msg)
    print(filename)
    fig.savefig(filename)
    return


def histogram_comparison_save(matrix, title, labels, params, n, msg=''):
    fig, ax = plt.subplots()
    ax.hist(matrix, bins= 32, stacked=False, histtype='bar', label=labels)
    ax.legend(prop={'size': 10})
    ax.set_title(title)

    time = strftime("%m-%d_%H-%M-%S", gmtime())
    filename = '../plots/heatmaps/Histogram_n={}_time_{}_beta={}_r={}{}.png'.format(n, time, params[0], params[1], msg)
    print(filename)
    fig.tight_layout()
    fig.savefig(filename)
    return


def visualize_regions():
    halves = 0.5*np.array([1, 1])
    axes = np.array([0, 1])
    plt.plot(halves, axes, color='k')
    plt.plot(axes, halves, color='k')

    def f(xx):
        if xx < 1/2:
            return 0.5 - xx
        elif xx < 3/4:
            return xx - 0.5
        elif xx < 1:
            return np.power((1 - np.sqrt(1-xx)), 2)
    detection_bound = np.vectorize(f)
    x = np.linspace(0, 1, 1000)
    y = detection_bound(x)
    plt.fill_between(x, y, np.zeros(1000), facecolor='b')
    plt.plot(x, y, color='r')
    plt.show()
    return

def visualize_regions_all():

    plt.xlabel(r'$\beta$')
    plt.ylabel(r'$r$')
    plt.title('Regions in phase diagram')
    plt.grid(color='silver', linestyle='--')
    plt.axis([0, 1, 0, 4])
    halves = 0.5*np.array([1, 1])
    axes = np.array([0, 1])
    plt.text(0.5, 3.5, 'Exactly recoverable')
    plt.text(0.2, 1.5, 'Almost fully recoverable')
    plt.text(0.4, 0.35, 'Not recoverable but detectable')
    plt.text(0.8, 0.1, 'Undetectable')


    def exrec(xx):
        return np.power((1 + np.sqrt(1 - xx)), 2)

    def alrec(xx):
        return xx

    def f(xx):
        if xx < 1/2:
            return 0 + xx - xx
        elif xx < 3/4:
            return xx - 0.5
        elif xx < 1:
            return np.power((1 - np.sqrt(1-xx)), 2)

    detection_bound = np.vectorize(f)
    exactly_recoverable = np.vectorize(exrec)
    almost_recoverable = np.vectorize(alrec)

    #trans = mtransforms.blended_transform_factory(ax.transData, ax.transAxes)

    x = np.linspace(0, 1, 1000)
    y_detection = detection_bound(x)
    #plt.fill_between(x, y, np.zeros(1000), facecolor='b')
    plt.plot(x, y_detection, color='r')
    plt.fill_between(x, 0, y_detection, facecolor='red', alpha=0.5)#, transform=trans)

    y_almost = almost_recoverable(x)
    plt.plot(x, y_almost, color='y', linestyle='-.')
    plt.fill_between(x, y_detection, y_almost, facecolor='yellow', alpha=0.5)#, transform=trans)


    y_exactly = exactly_recoverable(x)
    plt.plot(x, y_exactly, color='g')
    plt.fill_between(x, y_almost, y_exactly, facecolor='#C5FF33', alpha=0.5)#, transform=trans)


    plt.fill_between(x, y_exactly, 4, facecolor='green', alpha=0.5)#, transform=trans)

    plt.plot(axes, np.array([0, 0]), color='k')
    plt.plot(np.array([0, 0]), np.array([0, 4]), color='k')

    #ax.fill_between(x, 0, 1, where=y > theta, facecolor='green', alpha=0.5, transform=trans)
    #ax.fill_between(x, 0, 1, where=y > theta, facecolor='green', alpha=0.5, transform=trans)
    #ax.fill_between(x, 0, 1, where=y > theta, facecolor='green', alpha=0.5, transform=trans)


    plt.show()
    return


def visualize_regions_theta(thetas):
    plt.xlabel(r'$\beta$')
    plt.ylabel(r'$r$')
    plt.title('Theta = (0, 0.2, 0.4)')
    plt.grid(color='silver', linestyle='--')
    plt.axis([0, 1, 0, 1])

    x = np.linspace(0, 1, 1001)

    def alrec(xx):
        return xx

    def alrec_third(xx):
        if xx<0.75:
            return xx/3
        else:
            return np.nan
    almost_recoverable = np.vectorize(alrec)
    almost_recoverable_third = np.vectorize(alrec_third)
    y_almost = almost_recoverable(x)
    y_almost_third = almost_recoverable_third(x)
    plt.plot(x, y_almost, color='grey', linestyle='--' )
    plt.plot(x, y_almost_third, color='grey', linestyle='--' )

    for theta in thetas:
        def f(xx):
            if xx <= 1 / 2:
                return np.nan
            elif xx <= 3 / 4:
                return (1 - theta)*(xx - 0.5)
            elif xx <= 1:
                return (1 - theta)*np.power((1 - np.sqrt(1 - xx)), 2)

        detection_bound = np.vectorize(f)
        y = detection_bound(x/(1 - theta))
        print(y)
        plt.plot(x, y)

    plt.show()
    return


def visualize_regions_sigma(theta): # INTE IMPLEMENTERAD ÄN
    halves = 0.5 * np.array([1, 1])
    axes = np.array([0, 1])
    plt.plot(halves, axes, color='k')
    plt.plot(axes, halves, color='k')
    return



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


def hc_cscshm_1(x, beta, r, alpha=0.5):
    #  Csörgós Csörgós Horvath Mason
    n = x.shape[0]
    pi = calculate_p_values(x)
    sorted_pi, _ = sort_by_size(pi)

    n_float = float(n)
    """"  For-loop version of HC:
    alpha_0 = 1.0
    trunc = np.floor(alpha_0 * n).astype(int)
    for i in range(0, trunc):
        if sorted_pi[i] > 1/n:
            pi_val = sorted_pi[i]
            norm = pi_val * (1 - pi_val)
            hc_vector[i] = np.sqrt(n_float) * ((i+1)/n_float - pi_val) / np.sqrt(norm * np.log(np.log(1 / norm)))
    i_opt = np.argmax(hc_vector)
    hc_opt = np.max(hc_vector)
    pi_opt = sorted_pi[i_opt]
    opt_index = np.where(pi == pi_opt)
    z_opt = x[opt_index]
    print('For-loop: pi_opt', pi_opt, 'i_opt', i_opt, 'hc_opt', hc_opt, 'opt_index', opt_index, 'zopt', z_opt)
    # """

    # """ Vectorized version
    ii = (1 + np.arange(n_float, dtype=float)) / (n_float + 1)
    norm_alt = sorted_pi - np.power(sorted_pi, 2)
    hc_alt = np.sqrt(n_float) * np.divide((ii - sorted_pi), np.sqrt(norm_alt * np.log(np.log(1 / norm_alt))))

    #ind = np.where(sorted_pi <= 1 / n_float)
    #hc_alt[ind] = 0
    hc_alt = hc_alt[0:np.floor(alpha * n_float).astype(int)]

    max_hc_ind = np.argmax(hc_alt)
    opt_z_index = np.where(pi == sorted_pi[max_hc_ind])
    z_opt = x[opt_z_index]
    pi_opt = sorted_pi[max_hc_ind]
    hc_opt = np.max(hc_alt)
    #print('Vectorized: pi_opt', pi_opt, 'max_hc_ind', max_hc_ind, 'hc_opt', hc_opt, 'opt_index', opt_z_index, 'zopt', z_opt)
    # """
    save_figures(x, sorted_pi, hc_alt, hc_opt, max_hc_ind, beta, r,  alpha, 'CsCsHM1')

    #print('Optimal HC:', hc_opt, 'index i_opt:', i_opt)
    return max_hc_ind, hc_opt


def hc_cscshm_2(x, beta, r, alpha=0.5):
    #  Csörgós Csörgós Horvath Mason
    n = x.shape[0]
    pi = calculate_p_values(x)
    sorted_pi, _ = sort_by_size(pi)
    hc_vector = np.zeros(n)
    n_float = float(n)
    """ For-loop version
    alpha_0 = 0.5
    trunc = np.floor(alpha_0 * n).astype(int)
    for i in range(1, trunc):
        if sorted_pi[i] > 1/np.power(n, 2):
            pi_val = sorted_pi[i]
            norm = pi_val * (1 - pi_val)
            hc_vector[i] = np.sqrt(n_float) * (i/n_float - pi_val) / (np.sqrt(norm) * np.log(np.log(1 / norm)))
    i_opt = np.argmax(hc_vector)
    hc_opt = np.max(hc_vector)
    pi_opt = sorted_pi[i_opt]
    opt_index = np.where(pi == pi_opt)
    z_opt = x[opt_index]
    print('For-loop: pi_opt', pi_opt, 'i_opt', i_opt, 'hc_opt', hc_opt, 'opt_index', opt_index, 'zopt', z_opt)
    # """

    # """ Vectorized version
    ii = (1 + np.arange(n_float, dtype=float)) / (n_float + 1)
    norm_alt = sorted_pi - np.power(sorted_pi, 2)
    hc_alt = np.sqrt(n_float) * np.divide((ii - sorted_pi), np.sqrt(norm_alt) * np.log(np.log(1 / norm_alt)))

    #ind = np.where(sorted_pi <= 1 / n_float)
    #hc_alt[ind] = 0

    hc_alt = hc_alt[0:np.floor(alpha * n_float).astype(int)]

    max_hc_ind = np.argmax(hc_alt)
    opt_z_index = np.where(pi == sorted_pi[max_hc_ind])
    z_opt = x[opt_z_index]
    pi_opt = sorted_pi[max_hc_ind]
    hc_opt = np.max(hc_alt)
    #print('Vectorized: pi_opt', pi_opt, 'max_hc_ind', max_hc_ind, 'hc_opt', hc_opt, 'opt_index', opt_z_index, 'zopt',   z_opt)
    # """
    #print('Optimal HC:', hc_opt, 'index i_opt:', i_opt)
    save_figures(x, sorted_pi, hc_alt, hc_opt, max_hc_ind, beta, r,  alpha, 'CsCsHM2')

    return max_hc_ind, hc_opt


def hc_plus(x, beta, r, alpha=0.5, orthodox=0):
    n = x.shape[0]
    pi = calculate_p_values(x)
    sorted_pi, _ = sort_by_size(pi)
    hc_vector = np.zeros(n)
    n_float = float(n)
    """ for -loop
    alpha_0 = 0.5
    trunc = np.floor(alpha_0 * n).astype(int)
    for i in range(1, trunc):
        if sorted_pi[i] > 1/n:
            hc_vector[i] = np.sqrt(n_float) * (i/n_float - sorted_pi[i]) / np.sqrt(sorted_pi[i]*(1 - sorted_pi[i]))

    i_opt = np.argmax(hc_vector)
    hc_opt = np.max(hc_vector)
    pi_opt = sorted_pi[i_opt]
    opt_index = np.where(pi == pi_opt)
    z_opt = x[opt_index]
    print('For-loop: pi_opt', pi_opt, 'i_opt', i_opt, 'hc_opt', hc_opt, 'opt_index', opt_index, 'zopt', z_opt)
    # """

    ind = np.where(sorted_pi <= 1 / n_float)
    ii = (1 + np.arange(n_float)) / (n_float + 1)

    #hc_vector = np.sqrt(n_float) * (ii - sorted_pi) / np.sqrt(sorted_pi - np.power(sorted_pi, 2))

    hc_vector = np.sqrt(n_float) * (ii - sorted_pi) / np.sqrt(ii - np.power(ii, 2))

    if orthodox==0:
        hc_vector[ind] = 0

    hc_vector = hc_vector[0:np.floor(alpha * n_float).astype(int)]

    max_hc_ind = np.argmax(hc_vector)
    opt_z_index = np.where(pi == sorted_pi[max_hc_ind])
    #z_opt = x[opt_z_index]
    #pi_opt = sorted_pi[max_hc_ind]
    hc_opt = hc_vector[max_hc_ind]
    #opt_index = opt_z_index
    #print('Vectorized: pi_opt', pi_opt, 'max_hc_ind', max_hc_ind, 'hc_opt', hc_opt, 'opt_index', opt_index, 'zopt',z_opt)
    save_figures(x, sorted_pi, hc_vector, hc_opt, max_hc_ind, beta, r, alpha,  'HC_plus')

    #print('Optimal HC:', hc_opt, 'index i_opt:', i_opt)
    return max_hc_ind, hc_opt


def calculate_p_values(x):
    p_values = norm.sf(x)
    return p_values


def sort_by_size(pi):
    sort_index = np.argsort(pi)
    pi = np.sort(pi)
    return pi, sort_index