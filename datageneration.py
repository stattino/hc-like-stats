import numpy as np
from scipy.stats import chi2


def generate_data(n, beta, r, signal_presence, dist):
    if dist[1] == 'norm':
        # dist['var'] for later implementation?
        return generate_normal_mixture(n, beta, r, signal_presence)
    elif dist[1] == 'chi2':
        return generate_chi2_mixture(n, beta, r, signal_presence, dist['df'], dist['loc'])
    else:
        print('check distribution')

def generate_classification_data(n, p, beta, r, balance=0.5):
    if p < 2*n:
        print('...not considering p>>n...')
    tau = np.sqrt(2 * r * np.log(p))
    epsilon = np.power(p, -beta)
    mu0 = tau/np.sqrt(n)

    n_signals = np.ceil(epsilon * p)
    index = np.random.choice(p, n_signals, False)

    x = np.zeros([n, p])
    y = np.ones(n)
    n_class_2 = np.ceil(balance*n)
    index_class_2 = np.random.choice(n, n_class_2, False)
    y[index_class_2] *= -1

    for i in range(0, n):
        x[i, ] = np.random.normal(0, 1, p)
        x[i, index] = np.random.normal(mu0*y[i], 1, n_signals)

    return x, y


def generate_normal_mixture(n, beta, r, signal_presence):
    epsilon = np.power(n, -beta)
    if beta > 0.5:
        mu0 = np.power(2*r*np.log(n), 0.5)
    else:
        mu0 = np.power(n, -r)
    x = np.random.normal(0, 1, n)
    if signal_presence == 1:
        n_signals = np.ceil(epsilon*n)
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
        n_signals = np.ceil(epsilon*n)
        index = np.random.choice(n, n_signals, False)
        x[index] = chi2.rvs(df, location+w0, 1, n_signals)
    return x


def normalize_matrix(x):
    if len(x.shape) > 1:
        n, p = x.shape
    else:
        print('normalize_matrix requires matrix')
        return
    for col in range(0, p):
        mean = sum(x[:, col]/n)
        print(mean)
        variance = sum(np.power(np.add(x[:, col], -mean), 2))

        x[:, col] -= mean
        x[:, col] /= variance

    print(x)
    return
