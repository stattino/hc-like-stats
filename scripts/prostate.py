from detectionboundary import *
from highercrit import *

X = np.loadtxt('../../data/prostate/prostate.x.txt')
Y = np.loadtxt('../../data/prostate/prostate.y.txt')

# IF Z_type = 1 => need to normalize the data. ELSE don't.
#X = X - np.transpose(np.tile(np.mean(X, 1), (102, 1)))
#X = np.divide(X, np.transpose(np.tile(np.std(X, 1), (102, 1))))
z_type = 0
Y_cs = np.copy(Y)

# convert Y_train 0:s to -1:s
for i in range(0, Y.shape[0]):
    if Y[i] == 0:
        Y[i] = -1

# Since cscshm is dependent on which is bigger invert the labels
for i in range(0, Y.shape[0]):
    if Y_cs[i] == 0:
        Y_cs[i] = 1
    else:
        Y_cs[i] = 0

mean_mean = np.mean(X, 0)
std_std = np.std(X, 0)
std_other = np.std(X, 1)
mean_other = np.mean(X, 1)
# Create a M-fold random split of the 102 cases
N = X.shape[1]
M = 3
index = np.random.choice(np.arange(N), N, False)
chunk_size = int(np.floor(N / M))

error_hc = np.zeros(M)
error_cs1 = np.zeros(M)
error_cs2 = np.zeros(M)

# For every split, calculate the missclassification rate, and repeat M_2 times.
M_2 = 10
for j in range(0, M_2):
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

        error_hc[i] += sum(y_attempt != y_test) / y_test.shape
        #print('Error hc= ', error_hc[i], 'size', y_test.shape, 'total errors:', sum(y_attempt != y_test))

        y_test = Y_cs[test_idx]
        # CSCSHM
        selected, mu1, mu2, std = cscshm_thresholding(np.transpose(X[:, train_idx]), np.transpose(Y_cs[train_idx]), 0, 0, 1, z_type)
        y_attempt = cscshm_discriminant_rule(np.transpose(X[:, test_idx]), selected, mu1, mu2, std)

        error_cs1[i] += sum(y_attempt != y_test) / y_test.shape
        #print('Error cs1= ', error_cs1[i], 'size', y_test.shape, 'total errors:', sum(y_attempt != y_test))

        selected, mu1, mu2, std = cscshm_thresholding(np.transpose(X[:, train_idx]), np.transpose(Y_cs[train_idx]), 0, 0, 2, z_type)
        y_attempt = cscshm_discriminant_rule(np.transpose(X[:, test_idx]), selected, mu1, mu2, std)

        error_cs2[i] += sum(y_attempt != y_test) / y_test.shape
        #print('Error cs2= ', error_cs2[i], 'size', y_test.shape, 'total errors:', sum(y_attempt != y_test))


total_error_hc = np.divide(np.sum(error_hc), M*M_2)
total_error_cs1 = np.divide(np.sum(error_cs1), M*M_2)
total_error_cs2 = np.divide(np.sum(error_cs2), M*M_2)
print('tot error hc:', total_error_hc)
print('tot error cscshm1:', total_error_cs1)
print('tot error cscshm2:', total_error_cs2)
