import numpy as np
from datageneration import *
from highercrit import *
from datageneration import *

n = 100
p = 1000
r = 0.5
beta = 0.5
threshold = 'clip'
theta = 0.4

# x = np.array([[1, 2, 3], [1, -2, 4]], 'double')
# print(x)
# normalize_matrix(x)

x, y = generate_classification_data(p, theta, beta, r)

z_opt, weights = hc_thresholding(x, y, beta, r, threshold)

y_attempt = discriminant_rule(weights, x)

# print(weights)

# print(y_attempt==y)
# print(y)