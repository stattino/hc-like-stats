import numpy as np
from highercrit import *
from datageneration import *

n = 100
p = 10000
r = 0.5
beta = 0.5
threshold = 'clip'
theta = 0.25

# x = np.array([[1, 2, 3], [1, -2, 4]], 'double')
# print(x)
# normalize_matrix(x)
diff_1 = 0
diff_2 = 0
mm = 50

for i in range(0, mm):
    x_train, y_train, x_test, y_test = generate_classification_data(p, theta, beta, r)
    z_opt, weights = hc_thresholding(x_train, y_train, beta, r, hc_plus_classification, threshold)
    y_attempt = discriminant_rule(weights, x_test)

    error = sum(y_attempt!=y_test)/y_test.shape
    print('Error HCT = ', error, 'size',y_test.shape, 'total errors:', sum(y_attempt!=y_test))

    diff_1 -= sum(y_attempt!=y_test)
    diff_2 -= sum(y_attempt != y_test)

    x_train, y_train, x_test, y_test = generate_classification_data_cscshm(p, theta, beta, r)
    selected, mu1, mu2 = cscshm_thresholding(x_train, y_train, beta, r, 1)
    y_attempt = cscshm_discriminant_rule(x_test, selected, mu1, mu2)
    error = sum(y_attempt!=y_test)/y_test.shape

    print('Error CsCsHM1 = ', error, 'size',y_test.shape, 'total errors:', sum(y_attempt!=y_test))
    diff_1 += sum(y_attempt!=y_test)

    selected, mu1, mu2 = cscshm_thresholding(x_train, y_train, beta, r, 2)
    y_attempt = cscshm_discriminant_rule(x_test, selected, mu1, mu2)

    y_test = y_test.astype(int)

    error = sum(y_attempt!=y_test)/y_test.shape
    diff_2 += sum(y_attempt != y_test)

    print('Error CsCsHM2 = ', error, 'size',y_test.shape, 'total errors:', sum(y_attempt!=y_test))

print(" ---------( ----- )-----------")
print("Errors made by HCT vs cscshm1", diff_1, " % less good:", diff_1/(y_test.shape[0]*mm))
print("Errors made by HCT vs cscshm2", diff_2, " % less good:", diff_2/(y_test.shape[0]*mm))


print(mu1*selected)
print(mu2)
# print(y)
