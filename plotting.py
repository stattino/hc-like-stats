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


def heat_map_alt(matrix, n, msg):
    fig = plt.figure()
    nx, ny = matrix.shape
    if nx == ny:
        x = np.linspace(0, 0.5, nx+1)
        y = np.linspace(0, 0.5, ny+1)
        xv, yv = np.meshgrid(x, y)
        plt.pcolormesh(xv, yv, matrix.transpose(), cmap='RdYlBu')
        plt.xlim(0, 0.5)
        plt.ylim(0, 0.5)
    else:
        nx, ny = matrix.shape
        x = np.linspace(0.5, 1, nx+1)
        y = np.linspace(0, 1, ny+1)
        xv, yv = np.meshgrid(x, y)
        plt.pcolormesh(xv, yv, matrix.transpose(), cmap='RdYlBu')  #seismic, RdYlBu, coolwarm
        plt.xlim(0.5, 1)
        plt.ylim(0, 1)
    plt.colorbar()
    time = strftime("%m-%d_%H-%M-%S", gmtime())
    filename = 'plots/heatmaps/Detection_Boundary_n={}_grid={}x{}_time_{}_{}.png'.format(n, nx, ny, time, msg)
    print(filename)
    fig.savefig(filename)
    return


def heat_map_save(matrix, n, msg):
    time = strftime("%m-%d_%H-%M-%S", gmtime())
    filename = 'data/heatmap_data_{}_n={}_time_{}.txt'.format(msg, n, time)
    print(filename)
    np.savetxt(filename, matrix)
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