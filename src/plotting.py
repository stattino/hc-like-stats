import numpy as np
import matplotlib.pyplot as plt
from time import gmtime, strftime


def save_figures(z, pi, hcs, hc_opt, i_opt, beta, r, msg):
    alpha_0 = 0.1
    time = strftime("%m-%d_%H-%M-%S", gmtime())
    n = pi.shape[0]
    if n<1000000:
        x_points = np.arange(0, alpha_0, 1 / n)
    else:
        x_points = np.arange(0, alpha_0*n, 1)
    z = np.sort(z)
    z = z[::-1]
    fig = plt.figure()
    #print(x_points.shape, z.shape)
    plt.subplot(3, 1, 1)  # Z-scores
    plt.plot(x_points, z[0:alpha_0*n])
    plt.axvline(x=i_opt/n, ymax=hc_opt, color='#d62728')
    plt.axhline(y=z[i_opt], xmin=0, xmax=i_opt / (n*alpha_0), color='#d62728')

    plt.subplot(3, 1, 2)  # P-values
    plt.plot(x_points, pi[0:alpha_0*n])
    plt.axvline(x=i_opt / n, ymax=hc_opt, color='#d62728')

    plt.subplot(3, 1, 3)  # HC-statistic
    plt.plot(x_points, hcs[0:alpha_0*n])
    plt.axvline(x=i_opt / n, ymax=hc_opt, color='#d62728')
    plt.axhline(y=hc_opt, xmin=0, xmax=i_opt / (n*alpha_0), color='#d62728')

    filename = 'plots/hctest/{}-N={}-beta={}-r={}_{}.png'.format(msg, n, beta, r, time)
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
    ax.hist(matrix, bins= int (n/100), stacked=False, histtype='bar', label=labels)
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

    x = np.linspace(0, 1, 1000)
    y = detection_bound(x)
    #plt.fill_between(x, y, np.zeros(1000), facecolor='b')
    plt.plot(x, y, color='b')

    y = exactly_recoverable(x)
    plt.plot(x, y, color='r')

    y = almost_recoverable(x)
    plt.plot(x, y, color='g', linestyle='-.')
    plt.plot(axes, np.array([0, 0]), color='k')
    plt.plot(np.array([0, 0]), np.array([0, 4]), color='k')
    plt.show()
    return


def visualize_regions_theta(theta):
    halves = 0.5 * np.array([1, 1])
    axes = np.array([0, 1])
    plt.plot(halves, axes, color='k')
    plt.plot(axes, halves, color='k')

    def f(xx):
        if xx < 1 / 2:
            return xx-xx
        elif xx < 3 / 4:
            return (1 - theta)*(xx - 0.5)
        elif xx < 1:
            return (1 - theta)*np.power((1 - np.sqrt(1 - xx)), 2)

    detection_bound = np.vectorize(f)
    x = np.linspace(0, 1, 1000)
    y = detection_bound(x/(1 - theta))
    # plt.fill_between(x, y, np.zeros(1000), facecolor='b')
    plt.plot(x, y, color='r')
    plt.show()
    return


def visualize_regions_sigma(theta): # INTE IMPLEMENTERAD ÄN
    halves = 0.5 * np.array([1, 1])
    axes = np.array([0, 1])
    plt.plot(halves, axes, color='k')
    plt.plot(axes, halves, color='k')

    def f(xx):
        if xx < 1 / 2:
            return 0.5 - xx
        elif xx < 3 / 4:
            return xx - 0.5
        elif xx < 1:
            return np.power((1 - np.sqrt(1 - xx)), 2)

    detection_bound = np.vectorize(f)
    x = np.linspace(0, 1, 1000)
    y = detection_bound(x)
    plt.fill_between(x, y, np.zeros(1000), facecolor='b')
    plt.plot(x, y, color='r')
    plt.show()
    return