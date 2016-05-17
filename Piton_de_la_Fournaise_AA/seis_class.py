# -*- coding: utf-8 -*-
import numpy as np
import warnings
from pickle import load
from sklearn.cross_validation import cross_val_score, KFold
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from os.path import join

figdir = 'figure'

def clean_datasets(X_filename, Y_filename):

    # read data
    f_ = open(X_filename, 'r')
    X_orig, att_names = load(f_) 
    f_.close()

    f_ = open(Y_filename, 'r')
    Y_orig, Ydict_orig = load(f_)
    f_.close()


    n_ev, n_att = X_orig.shape


    # find keys for Effondrement and Sommital
    for key, value in Ydict_orig.iteritems(): 
        if value == 'Effondrement' : 
            eff_key = key
        elif value == 'Sommital' :
            som_key = key

    Y_clean = np.zeros(n_ev, dtype=int)
    for i in xrange(n_ev):
        if Y_orig[i] == eff_key or Y_orig[i] == som_key :
            Y_clean[i] = 1

    X = X_orig[:, :][Y_clean == 1]
    Y = Y_orig[:][Y_clean == 1]

    Ydict = {}
    Ydict[eff_key] = 'Effondrement'
    Ydict[som_key] = 'Sommital'

    # take logs of :
    for att in ['E', 'A', 'K']:
        i1 = att_names.index(att)
        i2 = i1 + n_att/2
        # check that Kurtosis is above zero. If not, then Fisher's 
        # definition was used. Add 3 to get to Pearson's kurtosis
        if att=='K' and np.min(X[:, [i1,i2]]) < 0.0 :
            X[:, [i1,i2]] = np.log10(X[:, [i1,i2]]+3.0)
        else:
            X[:, [i1,i2]] = np.log10(X[:, [i1,i2]])

    return X, att_names, Y, Ydict

def clean_datasets_split(X_filename, Y_filename):

      # read data
    f_ = open(X_filename, 'r')
    X_orig, att_names = load(f_) 
    f_.close()
    
    f_ = open(Y_filename, 'r')
    Y_orig, Ydict_orig = load(f_)
    f_.close()
    
    n_ev, n_att = X_orig.shape 
    
    X_eff=np.zeros(n_ev, dtype=int)
    X_som=np.zeros(n_ev, dtype=int)
    
    # find keys for Effondrement and Sommital
    for key, value in Ydict_orig.iteritems(): 
        if value == 'Effondrement' : 
            eff_key = key
        elif value == 'Sommital' :
            som_key = key
    
    
    Y_clean = np.zeros(n_ev, dtype=int)
    for i in xrange(n_ev):
        if Y_orig[i] == eff_key :
            Y_clean[i] = 1
        elif Y_orig[i] == som_key :
            Y_clean[i] = 2
    
    X_eff = X_orig[:, :][Y_clean == 1] # 121 for vf
    X_som = X_orig[:, :][Y_clean == 2] # 2171 for vf
    
    Ydict = {}
    Ydict[eff_key] = 'Effondrement'
    Ydict[som_key] = 'Sommital'

    # take logs of :
    for att in ['E', 'A', 'K']:
        i1 = att_names.index(att)
        i2 = i1 + n_att/2
        # check that Kurtosis is above zero. If not, then Fisher's 
        # definition was used. Add 3 to get to Pearson's kurtosis
        if att=='K' and np.min(X_eff[:, [i1,i2]]) < 0.0 :
            X_eff[:, [i1,i2]] = np.log10(X_eff[:, [i1,i2]]+3.0)
        else:
            X_eff[:, [i1,i2]] = np.log10(X_eff[:, [i1,i2]])
            
    for att in ['E', 'A', 'K']:
        i1 = att_names.index(att)
        i2 = i1 + n_att/2

        if att=='K' and np.min(X_som[:, [i1,i2]]) < 0.0 :
            X_som[:, [i1,i2]] = np.log10(X_som[:, [i1,i2]]+3.0)
        else:
            X_som[:, [i1,i2]] = np.log10(X_som[:, [i1,i2]])
            

    return X_eff, X_som, att_names, n_ev, n_att, Y_orig, Y_clean


def evaluate_cross_validation(clf, X, Y, k):
    # create a k-fold cross-validation iterator of k-folds
    cv = KFold(len(Y), k, shuffle=True)
    scores = cross_val_score(clf, X, Y, cv=cv)
    #print "Mean score: %.3f +/- %.3f"%(np.mean(scores), np.std(scores))
    print "%.3f"%(np.mean(scores))
    print "%.3f"%(np.std(scores))
    return np.mean(scores), np.std(scores)


def equalize_classes(X, Y, n_max=None):
    """
    Takes an X matrix and Y vector in which the classes are unbalanced, and
    returns an X matrix and Y vector with the same number of elements in each
    class. This number is given by the smallest class or n_max, if it is set. 
    Also returns the lines in X and Y that were not selected.
    """
    # isolate the classes
    classes = np.unique(Y)
    n_class = len(classes)

    # get the number of events in each class
    n_per_class = np.empty(n_class, dtype=int)
    for i in xrange(n_class):
        n_per_class[i] = len(Y[Y==classes[i]])
    n_min = np.min(n_per_class)
    if n_max is not None:
        n_min = np.min([n_min, n_max])


    # do extraction
    X_tmp_list = []
    Y_tmp_list = []
    X_test_list = []
    Y_test_list = []
        
    for i in xrange(n_class):
        indexes = np.random.permutation(n_per_class[i])
        Xi = X[:, :][Y==classes[i]]
        Yi = Y[:][Y==classes[i]]
        X_tmp_list.append(Xi[indexes[0:n_min], :])
        Y_tmp_list.append(Yi[0:n_min])
        X_test_list.append(Xi[indexes[n_min:], :])
        Y_test_list.append(Yi[n_min:])
    X_uni = np.vstack(X_tmp_list)
    Y_uni = np.hstack(Y_tmp_list)
    X_test = np.vstack(X_test_list)
    Y_test = np.hstack(Y_test_list)

    return X_uni, Y_uni, X_test, Y_test

def train_and_evaluate(clf, X_train, X_test, Y_train, Y_test):
    clf.fit(X_train, Y_train)
    print "Accuracy on training set: %.2f"%clf.score(X_train, Y_train)
    print "Accuracy on testing set: %.2f"%clf.score(X_test, Y_test)

    Y_pred = clf.predict(X_test)

    print "Confusion matrix:"
    cm = confusion_matrix(Y_test, Y_pred)
    print cm
    return cm, clf.score(X_train, Y_train), clf.score(X_test, Y_test)


def plot_confusion_matrix(cm, labels, title, filename):

    n_classes = len(labels)

    sums = np.sum(cm, axis=1)
    cm_norm = np.empty(cm.shape, dtype=float)
    for i in xrange(n_classes):
        cm_norm[i, :] = cm[i, :]/np.float(sums[i])*100.

    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.sca(ax)
    plt.matshow(cm_norm, cmap=plt.cm.gray_r)
    for i in xrange(n_classes):
        for j in xrange(n_classes):
            plt.text(j, i, "%d\n(%.1f%%)"%(cm[i, j], cm_norm[i, j]),
                     horizontalalignment='center', verticalalignment='center',
                     bbox=dict(facecolor='white', color='white'))
    plt.title(title)
    plt.colorbar(label='Accuracy (%)')
    plt.xticks(np.arange(len(labels)), labels)
    plt.yticks(np.arange(len(labels)), labels)
    plt.xlabel('Predicted')
    plt.ylabel('True')

    plt.savefig(join(figdir, filename))
    plt.close()


if __name__ == '__main__':

    X, att_names, Y, Ydict = clean_datasets('X.dat', 'Y.dat')
