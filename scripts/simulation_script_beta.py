from heatmapthresholding import *
from detectionboundary import normalize_colors, heat_map_save
from plotting import *


M = 20
K = 10
r = 0.8
p = 1000
theta = 0.3

alpha_cscshm = 0.3
alpha_hc = 1


error_hc = np.zeros((K, M))
error_cs1 = np.zeros((K, M))
error_cs2 = np.zeros((K, M))
no_var_hc = np.zeros((K, M))
no_var_cs1 = np.zeros((K, M))
no_var_cs2 = np.zeros((K, M))

beta_range = np.linspace(0, 1-theta, K)

for i in range(0, K):
    print('outer iteration', i+1, '/', K)
    beta = beta_range[i]
    for j in range(0, M):
        #HC_PLUS
        x_train, y_train, x_test, y_test = generate_classification_data(p, theta, beta, r)
        weights = hc_thresholding(x_train, y_train, alpha_hc)
        y_attempt = discriminant_rule(x_test, weights)

        error_hc[i, j] = sum(y_attempt != y_test) / y_test.shape
        no_var_hc[i, j] = sum(weights != 0)

        # CSCSHM_1 + CSCSHM_2
        x_train, y_train, x_test, y_test = generate_classification_data_cscshm(p, theta, beta, r)

        selected = cscshm_thresholding(x_train, y_train, 1, alpha_cscshm)
        y_attempt = cscshm_discriminant_rule(x_test, selected, r)
        error_cs1[i, j] = sum(y_attempt != y_test) / y_test.shape
        no_var_cs1[i, j] = sum(selected != 0)

        selected = cscshm_thresholding(x_train, y_train, 2, alpha_cscshm)
        y_attempt = cscshm_discriminant_rule(x_test, selected, r)
        error_cs2[i, j] = sum(y_attempt != y_test) / y_test.shape
        no_var_cs2[i, j] = sum(selected != 0)


mean_error_hc = np.sum(error_hc, 1) / M
mean_error_cs1 = np.sum(error_cs1, 1) / M
mean_error_cs2 = np.sum(error_cs2, 1) / M

std_error_hc = np.std(error_hc, 1)
std_error_cs1 = np.std(error_cs1, 1)
std_error_cs2 = np.std(error_cs2, 1)

mean_no_var_hc = np.sum(no_var_hc, 1) / M
mean_no_var_cs1 = np.sum(no_var_cs1, 1) / M
mean_no_var_cs2 = np.sum(no_var_cs2, 1) / M

std_no_var_hc = np.std(no_var_hc, 1)
std_no_var_cs1 = np.std(no_var_cs1, 1)
std_no_var_cs2 = np.std(no_var_cs2, 1)

# Visualize as a plot with STD-bars:

fig, axs = plt.subplots(nrows=2, ncols=1, sharex=True)
ax = axs[0]
ax.set_title('HCT vs. CsCsHM thresholding')

ax.errorbar(beta_range, mean_error_hc, xerr=0, yerr=std_error_hc,
            color='#3e3ef9', ecolor='#0909bf', capsize=4, fmt='o--', label=r'$HC$')
ax.errorbar(beta_range, mean_error_cs1, xerr=0, yerr=std_error_cs1,
  color='#ff3232', ecolor='#d31515', capsize=4, fmt='o--', label=r'$CsCsHM_1$')
#ax.errorbar(r_range, mean_error_cs2, xerr=0, yerr=std_error_cs2,
#            color='#ff3232', ecolor='#d31515', capsize=4, fmt='o--', label=r'$CsCsHM_2$')
#ax.set_title(r'Mean error')

ax.set_ylabel('Mean error')
ax.set_xlabel(r'$\beta$')
ax.grid(color='silver', linestyle='--')

handles, labels = ax.get_legend_handles_labels()
ax.legend(handles, labels)

ax = axs[1]
ax.errorbar(beta_range, mean_no_var_hc, xerr=0, yerr=std_no_var_hc,
            color='#3e3ef9', ecolor='#0909bf', capsize=4, fmt='o--', label=r'$HC$')

ax.errorbar(beta_range, mean_no_var_cs1, xerr=0, yerr=std_no_var_cs1,
  color='#ff3232', ecolor='#d31515', capsize=4, fmt='o--', label=r'$CsCsHM_1$')
#ax.errorbar(r_range, mean_no_var_cs2, xerr=0, yerr=std_no_var_cs2,
#            color='#ff3232', ecolor='#d31515', capsize=4, fmt='o--', label=r'$CsCsHM_2$')
#ax.set_title(r'Mean number of variables selected')

ax.set_ylabel('Mean n. variables')
ax.set_xlabel(r'$\beta$')
ax.grid(color='silver', linestyle='--')

handles, labels = ax.get_legend_handles_labels()
ax.legend(handles, labels)

filename = '../plots/heatmaps/Beta-Errorbars_M={}_K={}_p_{}_r={}_theta={}_alphacs={}_alphahc={}.png'.format(M, K, p, r, theta, alpha_cscshm, alpha_hc)
fig.savefig(filename)

plt.show()