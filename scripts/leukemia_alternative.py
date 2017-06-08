from realthresholding import *
import numpy as np

leukemia_x = np.loadtxt('../../data/leukemia/leukemia.x.txt');
leukemia_y = np.loadtxt('../../data/leukemia/leukemia.y.txt');
Data = leukemia_x
Class = leukemia_y


for i in range(0, leukemia_y.shape[0]):
    if leukemia_y[i] == 0:
        leukemia_y[i] = -1
    else:
        leukemia_y[i] = 1

P, N = Data.shape
z_type = 0

idx = np.random.permutation(72)

test_idx = idx[1:25]
train_idx =  idx[26:72]

train_data = Data[:, train_idx]
train_labels = Class[train_idx]

test_data = Data[:, test_idx]
test_labels = Class[test_idx]


z_opt, weights, mu1, mu2, std = hc_thresholding(np.transpose(train_data), np.transpose(train_labels))

y_attempt = discriminant_rule(weights, np.transpose(test_data), mu1, mu2, std)

# print error
#print(weights)
error = sum(y_attempt != test_labels) / test_labels.shape
print('Error= ', error, 'size', test_labels.shape, 'total errors:', sum(y_attempt != test_labels))
print('Nomber of weights: ', np.sum(weights!= 0), '/', weights.shape[0], ' class 1: ', np.sum(weights==1), ' class -1: ', np.sum(weights==-1))
print('----------------- //----------------- // -----------------')

sel_vec, mu1, mu2, sest = cscshm_thresholding(np.transpose(train_data), np.transpose(train_labels))
y_attempt = cscshm_discriminant_rule(np.transpose(test_data), sel_vec, mu1, mu2, sest)

zeroind = np.where(y_attempt == 0)
y_attempt[zeroind] = -1

error = sum(y_attempt != test_labels) / test_labels.shape
print('Error= ', error, 'size', test_labels.shape, 'total errors:', sum(y_attempt != test_labels))
print('Nomber of weights: ', np.sum(sel_vec!= 0), '/', sel_vec.shape[0])

"""
%%

[wts, stats] = HCclassification(train_data, train_labels, 'clip', 0.2);
%Number of useful features
HC_threshold = stats.HCT

nonzeroweights = sum(wts ~= 0)

%Run the data with HCclassification_fit function. Find the corresponding estimated labels and error rate

[label, score] = HCclassification_fit(wts, stats.xbar, stats.s, test_data);
HCerr = mean(label ~= test_labels)

g1 = find(test_labels == 1); g2 = find(test_labels == 0);
plot(g1, score(g1), 'ro', g2, score(g2), 'b+', 1:149, 0*(1:149), 'b--')
title('Classification Score for Test Data')
"""