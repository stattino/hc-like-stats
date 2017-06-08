from detectionboundary import *
import numpy as np

# Import the data
X = np.loadtxt('../../data/colon/colon.x.txt')
Y = np.loadtxt('../../data/colon/colon.y.txt')

mean_mean = np.mean(X, 0)
std_std = np.std(X, 0)
std_other = np.std(X, 1)
mean_other = np.mean(X, 1)
# whiten data
# CSCSHM FUNGERAR BÄTTRE MED Z_type = 1
#X = X - np.transpose(np.tile(np.mean(X, 1), (62, 1)))
#X = np.divide(X, np.transpose(np.tile(np.std(X, 1), (62, 1))))
z_type = 0


Y_cs = np.copy(Y)

# convert Y_train 0:s to -1:s
for i in range(0, Y.shape[0]):
    if Y[i] == 0:
        Y[i] = -1
"""
for i in range(0, Y.shape[0]):
    if Y_cs[i] == 0:
        Y_cs[i] = 1
    else:
        Y_cs[i] = 0
"""

# create M-fold random split
N = X.shape[1]
M = 10
M_2 = 10

error_hc = np.zeros((M_2, M))
error_cs1 = np.zeros((M_2, M))
error_cs2 = np.zeros((M_2, M))

no_var_hc = np.zeros((M_2, M))
no_var_cs1 = np.zeros((M_2, M))
no_var_cs2 = np.zeros((M_2, M))

# For every split, calculate the missclassification rate, and repeat M_2 times.
for j in range(0, M_2):
    index = np.random.choice(np.arange(N), N, False)
    chunk_size = int(np.floor(N / M))
    print("outer iteration", j)
    for i in range(0, M):
        if i < M - 1:
            test_idx = index[i * chunk_size: (i+1) * chunk_size]
        else:
            test_idx = index[i * chunk_size:]
        train_idx = np.delete(index, test_idx, 0)

        # HC_plus
        z_opt, weights, mu1, mu2, std = hc_thresholding(np.transpose(X[:, train_idx]), np.transpose(Y[train_idx]), 0, 0, 'default', 'hard', z_type)
        y_attempt = discriminant_rule(weights, np.transpose(X[:, test_idx]), mu1, mu2, std)
        y_test = Y[test_idx]

        error_hc[j, i] = sum(y_attempt != y_test) / y_test.shape
        no_var_hc[j, i] = sum(weights != 0)
        #print('Error hc= ', error_hc[i], 'size', y_test.shape, 'total errors:', sum(y_attempt != y_test))


        y_test = Y_cs[test_idx]
        # CSCSHM
        selected, mu1, mu2, std = cscshm_thresholding(np.transpose(X[:, train_idx]), np.transpose(Y_cs[train_idx]), 0, 0, 1, z_type)
        y_attempt = cscshm_discriminant_rule(np.transpose(X[:, test_idx]), selected, mu1, mu2, std)

        error_cs1[j, i] = sum(y_attempt != y_test) / y_test.shape
        no_var_cs1[j, i] = sum(selected != 0)
        #print('Error cs1= ', error_cs1[i], 'size', y_test.shape, 'total errors:', sum(y_attempt != y_test))

        selected, mu1, mu2, std = cscshm_thresholding(np.transpose(X[:, train_idx]), np.transpose(Y_cs[train_idx]), 0, 0, 2, z_type)
        y_attempt = cscshm_discriminant_rule(np.transpose(X[:, test_idx]), selected, mu1, mu2, std)

        error_cs2[j, i] = sum(y_attempt != y_test) / y_test.shape
        no_var_cs2[j, i] = sum(selected != 0)

        #print('Error cs2= ', error_cs2[i], 'size', y_test.shape, 'total errors:', sum(y_attempt != y_test))


mean_error_hc = np.sum(error_hc, 1) / M
mean_error_cs1 = np.sum(error_cs1, 1) / M
mean_error_cs2 = np.sum(error_cs2, 1) / M

est_error_hc = np.sum(mean_error_hc) / M_2
est_error_cs1 = np.sum(mean_error_cs1) / M_2
est_error_cs2 = np.sum(mean_error_cs2) / M_2

std_error_hc = np.std(mean_error_hc)
std_error_cs1 = np.std(mean_error_cs1)
std_error_cs2 = np.std(mean_error_cs2)

mean_no_var_hc = np.sum(no_var_hc, 1) / M
mean_no_var_cs1 = np.sum(no_var_cs1, 1) / M
mean_no_var_cs2 = np.sum(no_var_cs2, 1) / M

est_no_var_hc = np.sum(mean_no_var_hc) / M_2
est_no_var_cs1 = np.sum(mean_no_var_cs1) / M_2
est_no_var_cs2 = np.sum(mean_no_var_cs2) / M_2

print('tot error hc:', est_error_hc, ' +- ', std_error_hc)
print('tot error cscshm1:', est_error_cs1, ' +- ', std_error_cs1)
print('tot error cscshm2:', est_error_cs2, ' +- ', std_error_cs2)

print('Mean no. variables selected: HC ', est_no_var_hc, ' Cs_1 ', est_no_var_cs1, ' Cs_2 ', est_no_var_cs2)
print('Size of data:', X.shape)