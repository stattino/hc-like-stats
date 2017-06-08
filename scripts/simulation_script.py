from detectionboundary import *


M = 10
K = 10
beta = 0.5
p = 1000
theta = 0.5

error_hc = np.zeros((K, M))
error_cs1 = np.zeros((K, M))
error_cs2 = np.zeros((K, M))

no_var_hc = np.zeros((K, M))
no_var_cs1 = np.zeros((K, M))
no_var_cs2 = np.zeros((K, M))

r_range = np.linspace(0.5, 1, M)

for i in range(0, K):

    for j in range(0, M):
        #HC_PLUS
        x_train, y_train, x_test, y_test = generate_classification_data(p, theta, beta, r_range[j])
        _, weights,_, _, _ = hc_thresholding(x_train, y_train, beta, r_range[i], hc_plus_classification, 'clip', 1)
        y_attempt = old_discriminant_rule(weights, x_test)
        error_hc[i, j] = sum(y_attempt != y_test) / y_test.shape

        x_train, y_train, x_test, y_test = generate_classification_data_cscshm(p, theta, beta, r_range[j])
        # CSCSHM_1 + CSCSHM_2
        selected, mu1, mu2, std = cscshm_thresholding(x_train, y_train, 0, 0, 1, 1)
        y_attempt = cscshm_discriminant_rule(x_test, selected, mu1, mu2, std)
        error_cs1[i, j] = sum(y_attempt != y_test) / y_test.shape

        selected, mu1, mu2, std = cscshm_thresholding(x_train, y_train, 0, 0, 2, 1)
        y_attempt = cscshm_discriminant_rule(x_test, selected, mu1, mu2, std)
        error_cs2[i, j] = sum(y_attempt != y_test) / y_test.shape

print(np.mean(error_cs2, 0), np.mean(error_hc, 0))