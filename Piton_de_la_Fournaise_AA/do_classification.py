# -*- coding: utf-8 -*-
from seis_class import clean_datasets, evaluate_cross_validation
from seis_class import equalize_classes, train_and_evaluate,\
     plot_confusion_matrix
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from class_plot import plot_class_all 
from pickle import dump

X_filename = 'XEP.dat'
Y_filename = 'YEP.dat'
nmax = 500 # si change ici, change plus bas pour les plot 2 à 2
equalize_test_data = True

# read data and take only the chosen events 
X, att_names, Y, Ydict = clean_datasets(X_filename, Y_filename)
n_ev, n_att = X.shape
print X.shape
print Ydict

# 2550 som, 727 eff in all the catalogue
# 2926 som, 256 eff for vf

############################ STA1
# attributes to use
att_indexes = []
att_indexes.append(att_names.index('K'))
att_indexes.append(att_names.index('DUR'))
att_indexes.append(att_names.index('cent_f'))
att_indexes.append(att_names.index('A'))
att_indexes.append(att_names.index('dur/A'))
att_indexes.append(att_names.index('dom_f'))
att_indexes.append(att_names.index('E'))
att_indexes.append(att_names.index('maxA_mean'))
#att_indexes.append(att_names.index('AsDec'))

# STA 2
att_indexes.append(att_names.index('K')+n_att/2)
att_indexes.append(att_names.index('DUR')+n_att/2)
att_indexes.append(att_names.index('cent_f')+n_att/2)
att_indexes.append(att_names.index('A')+n_att/2)
att_indexes.append(att_names.index('dur/A')+n_att/2)
att_indexes.append(att_names.index('dom_f')+n_att/2)
att_indexes.append(att_names.index('E')+n_att/2)
att_indexes.append(att_names.index('maxA_mean')+n_att/2)
#att_indexes.append(att_names.index('AsDec')+n_att/2)

# clusterizer
clf1 = SVC(kernel='linear')
clf2 = SVC(kernel='rbf')

# dump the clf
f_ = open('clf1.dat', 'w')
dump(clf1, f_)
f_.close()

f_ = open('clf2.dat', 'w')
dump(clf2, f_)
f_.close()

#f_ = open('clf1.dat', 'w')
#dump((clf1.coef_[:], clf1.intercept_[:], clf1.support_vectors_[:]), f_)
#f_.close()
    
    
# reduce attributes
X = X[:, att_indexes]
print "Using the following %d attributes :"%len(att_indexes)
for i in att_indexes:
    print att_names[i]

# normalize data
scaler = MinMaxScaler().fit(X)

# equalize training data
X_train, Y_train, X_test, Y_test = equalize_classes(X, Y, nmax)
print X_train.shape
print X_test.shape

# equalize test data if requested
if equalize_test_data:
    X_tmp = X_test
    Y_tmp = Y_test
    X_test, Y_test, X_dump, Y_dump = equalize_classes(X_tmp, Y_tmp)
print X_test.shape




# do cross validation
print 'linear'
evaluate_cross_validation(clf1, X_train, Y_train, 5)
print 'rbf'
evaluate_cross_validation(clf2, X_train, Y_train, 5)

# create confusion matrix for test data
print 'linear'
cm1 = train_and_evaluate(clf1, X_train, X_test, Y_train, Y_test)
print 'rbf'
cm2 = train_and_evaluate(clf2, X_train, X_test, Y_train, Y_test)

# set the labels for the confusion matrix
labels = ['EFF','SOM']

# plot linear confusion matrix
title = 'Linear SVM'
filename = 'linear.png'
#plot_confusion_matrix(cm1, labels, title, filename)    

# plot rbf confusion matrix
title = 'rbf SVM'
filename = 'rbf.png'
#plot_confusion_matrix(cm2, labels, title, filename)   

################# plot attributs 

# for the training set
X_eff=X_train[0:500,:]
X_som=X_train[500:500*2,:]

n_attind=len(att_indexes)/2

"""
if n_attind == 8:
    plot_class_all(X_eff, X_som, n_attind)

"""

import matplotlib.pyplot as plt

n_attind = len(att_indexes)

    #station 1
"""
plt.scatter(X_eff[:,0],X_eff[:,1], c='r', marker='x') 
plt.scatter(X_som[:,0],X_som[:,1], c='w', marker='o', s=8) 
plt.title('RVL')
plt.xlabel('A')
plt.ylabel('Dur/A')

plt.savefig('A_DurA.png')
"""

