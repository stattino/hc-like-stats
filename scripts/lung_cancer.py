%% An example with Lung Cancer data
clc, clear all
%%
load('lungCancer.mat')

train_data = lungCancertrain(:, 1:(end-1));
train_class = lungCancertrain(:, end);

test_data = lungCancer_test(1:149, 1:12533);
test_class = lungCancer_test(1:149, 12534);

%%
test_data = test_data';
train_data = train_data';

%% Random test and train data

Data = [train_data, test_data];
Class = [train_class; test_class];
Data = Data';
[N, P] = size(Data);

%% Splits of data into training and testing
idx = randperm(N, N);

test_idx = idx(1:60);
train_idx =  idx(61:end);

train_data = Data(train_idx, :)';
train_class = Class(train_idx);

test_data = Data(test_idx, :)';
test_class = Class(test_idx);

%%

[wts, stats] = HCclassification(train_data, train_class, 'clip', 0.2);
%Number of useful features
HC_threshold = stats.HCT

nonzeroweights = sum(wts ~= 0)

%Run the data with HCclassification_fit function. Find the corresponding estimated labels and error rate

[label, score] = HCclassification_fit(wts, stats.xbar, stats.s, test_data);
HCerr = mean(label ~= test_class)

g1 = find(test_class == 1); g2 = find(test_class == 0);
plot(g1, score(g1), 'ro', g2, score(g2), 'b+', 1:149, 0*(1:149), 'b--')
title('Classification Score for Test Data')