import numpy as np
from datageneration import *
from highercrit import *
from datageneration import *

n = 100
p = 1000
r = 0.5
beta = 0.8
threshold = 'clip'
theta = 0.5

# x = np.array([[1, 2, 3], [1, -2, 4]], 'double')
# print(x)
# normalize_matrix(x)

x_train, y_train, x_test, y_test = generate_classification_data(p, theta, beta, r)

z_opt, weights = hc_thresholding(x_train, y_train, beta, r, threshold)

y_attempt = discriminant_rule(weights, x_test)

# print(weights)

error = sum(y_attempt!=y_test)/y_test.shape
print('Error= ', error, 'size',y_test.shape, 'total errors:', sum(y_attempt!=y_test))
# print(y)