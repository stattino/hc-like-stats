import os
from detectionboundary import *

# Dense grid creating:
N = 50
average_matrix = np.zeros((50, 50))
temp_matrix = np.zeros((50, 50))
counter = 0
for fn in os.listdir('../pdcresults'):
    if fn.endswith('.txt') and fn.startswith('heatmap_data_CsCsHM1'):
        #print(fn)
        string = '../pdcresults/' + fn
        temp_matrix = np.loadtxt(string)
        print(temp_matrix)
        average_matrix = np.add(average_matrix, temp_matrix)
        counter +=1
        temp_matrix = np.zeros((50, 50))

print(average_matrix)
average_matrix = np.divide(average_matrix, np.tile(counter, (50, 50)))

print(np.tile(counter, (50, 50)))
print(average_matrix)

normalize_colors(average_matrix)

x_lim = np.array([0, 0.5])
y_lim = np.array([0, 0.5])

heat_map_alt(average_matrix, 10000, x_lim, y_lim, 'CsCsHM2_dense')



