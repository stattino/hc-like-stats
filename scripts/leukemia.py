from detectionboundary import *
import numpy as np

# Import the data set
X_train = np.loadtxt('../../data/leukemia/ALL_vs_AML_train_set_38_matrix.txt')
X_test = np.loadtxt('../../data/leukemia/Leuk_ALL_AML_test_matrix.txt')

# whiten data
"""
X_train = X_train - np.transpose(np.tile(np.mean(X_train, 1), (38, 1)))
X_train = np.divide(X_train, np.transpose(np.tile(np.std(X_train, 1), (38, 1))))

X_test = X_test - np.transpose(np.tile(np.mean(X_test, 1), (35, 1)))
X_test = np.divide(X_test, np.transpose(np.tile(np.std(X_test, 1), (35, 1))))
"""
z_type = 0

# Create the labels
Y_train = np.concatenate([np.ones(27), np.multiply(np.ones(11),-1)])
Y_test = np.concatenate([np.ones(21), np.multiply(np.ones(14),-1)])


# Train the classifier on training data
z_opt, weights, mu1, mu2, std = hc_thresholding(np.transpose(X_train), np.transpose(Y_train), 0, 0, hc_plus_classification, 'hard')

# Test the classifier on the test data
y_attempt = discriminant_rule(weights, np.transpose(X_test), mu1, mu2, std)

# print error
error = sum(y_attempt != Y_test) / Y_test.shape
print('Error= ', error, 'size', Y_test.shape, 'total errors:', sum(y_attempt != Y_test))
print('------- ( ------ ) ----------')

X = np.append(X_train, X_test, 1)
Y = np.append(Y_train, Y_test)

mean_mean = np.mean(X, 0)
std_std = np.std(X, 0)
std_other = np.std(X, 1)
mean_other = np.mean(X, 1)

Y_cs = np.zeros(Y.shape)
for i in range(0, Y_cs.shape[0]):
    if Y[i] == -1:
        Y_cs[i] = 0
    else:
        Y_cs[i] = 1

# create M-fold random split
N = X.shape[1]
M = 10
index = np.random.choice(np.arange(N), N, False)
chunk_size = int(np.floor(N / M))

error = np.zeros(M)


error_hc = np.zeros(M)
error_cs1 = np.zeros(M)
error_cs2 = np.zeros(M)
# For every split, calculate the regret

M_2 = 20
for j in range(0, M_2):
    print("outer iteration", j+1)
    for i in range(0, M):
        if i < M - 1:
            test_idx = index[i * chunk_size: (i+1) * chunk_size]
        else:
            test_idx = index[i * chunk_size:]
        train_idx = np.delete(index, test_idx, 0)

        # Train the classifier
        z_opt, weights, mu1, mu2, std = hc_thresholding(np.transpose(X[:, train_idx]), np.transpose(Y[train_idx]), 0, 0, hc_plus, 'hard', z_type)
        y_attempt = discriminant_rule(weights, np.transpose(X[:, test_idx]), mu1, mu2, std)
        y_test = Y[test_idx]

        error_hc[i] += sum(y_attempt != y_test) / y_test.shape
        #print('Error hc= ', error_hc[i], 'size', y_test.shape, 'total errors:', sum(y_attempt != y_test))

        y_test = Y_cs[test_idx]

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