import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.feature_selection import SelectKBest, chi2
from synthetics.syncat_io import proc_syn_data
from graphics.graphics_class import plot_2D_scatter

# get matrices and attribute names
X, y, names = proc_syn_data()

#X_2D = SelectKBest(chi2, k=2).fit_transform(X, y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
#X_train_2D, X_test_2D, y_train_2D, y_test_2D = train_test_split(X_2D, y,
#                                                                test_size=0.25)

plot_2D_scatter(X_train[:, [0, 1]], X_test[:, [0, 1]], y_train, y_test,
                ['Gaussian', 'Exponential'], names[0], names[1], 'syn_2D_1.png')

plot_2D_scatter(X_train[:, [0, 2]], X_test[:, [0, 2]], y_train, y_test,
                ['Gaussian', 'Exponential'], names[0], names[2], 'syn_2D_2.png')

plot_2D_scatter(X_train[:, [0, 3]], X_test[:, [0, 3]], y_train, y_test,
                ['Gaussian', 'Exponential'], names[0], names[3], 'syn_2D_3.png')


#plot_2D_scatter(X_train2D, X_test_2D, y_train_2D, y_test_2D,
#                ['Gaussian', 'Exponential'], 'Attribute 1', 'Attribute 2',
#                'syn_2D_best.png')
