import numpy as np
from sklearn.cross_validation import train_test_split
from synthetics.syncat_io import proc_syn_data
from graphics.graphics_class import plot_2D_scatter

# get matrices and attribute names
X, y, names = proc_syn_data()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

plot_2D_scatter(X_train[:, [0, 1]], X_test[:, [0, 1]], y_train, y_test,
                ['Gaussian', 'Exponential'], names[0], names[1], 'syn_2D_1.png')

plot_2D_scatter(X_train[:, [0, 2]], X_test[:, [0, 2]], y_train, y_test,
                ['Gaussian', 'Exponential'], names[0], names[2], 'syn_2D_2.png')
