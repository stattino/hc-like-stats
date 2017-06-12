from realthresholding import *

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
z_opt, weights, mu1, mu2, std = hc_thresholding(np.transpose(X_train), np.transpose(Y_train))

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
        z_opt, weights, mu1, mu2, std = hc_thresholding(np.transpose(X[:, train_idx]), np.transpose(Y[train_idx]))
        y_attempt = discriminant_rule(weights, np.transpose(X[:, test_idx]), mu1, mu2, std)
        y_test = Y[test_idx]

        error_hc[j, i] = sum(y_attempt != y_test) / y_test.shape
        no_var_hc[j, i] = sum(weights != 0)
        #print('Error hc= ', error_hc[i], 'size', y_test.shape, 'total errors:', sum(y_attempt != y_test))


        y_test = Y_cs[test_idx]
        # CSCSHM
        selected, mu1, mu2, std = cscshm_thresholding(np.transpose(X[:, train_idx]), np.transpose(Y_cs[train_idx]), 1)
        y_attempt = cscshm_discriminant_rule(np.transpose(X[:, test_idx]), selected, mu1, mu2, std)

        error_cs1[j, i] = sum(y_attempt != y_test) / y_test.shape
        no_var_cs1[j, i] = sum(selected != 0)
        #print('Error cs1= ', error_cs1[i], 'size', y_test.shape, 'total errors:', sum(y_attempt != y_test))

        selected, mu1, mu2, std = cscshm_thresholding(np.transpose(X[:, train_idx]), np.transpose(Y_cs[train_idx]), 2)
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


std_no_var_hc = np.std(mean_no_var_hc)
std_no_var_cs1 = np.std(mean_no_var_cs1)
std_no_var_cs2 = np.std(mean_no_var_cs2)

print('---- HC plus ------------------------')
print('tot error:', est_error_hc, ' +- ', std_error_hc)
print('Mean no. variables selected: ', est_no_var_hc, ' +- ', std_no_var_hc)

print('---- CsCsHM 1 ------------------------')
print('tot error cscshm1:', est_error_cs1, ' +- ', std_error_cs1)
print('Mean no. variables selected: ', est_no_var_cs1, ' +- ', std_no_var_cs1)

print('---- CsCsHM 2 ------------------------')
print('tot error:', est_error_cs2, ' +- ', std_error_cs2)
print('Mean no. variables selected: ', est_no_var_cs2, ' +- ', std_no_var_cs2)

print('Size of data:', X.shape)