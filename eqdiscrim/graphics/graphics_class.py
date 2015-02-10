import numpy as np
import matplotlib.pyplot as plt
from os.path import join

colors = ['red', 'blue', 'green', 'cyan']
figdir = '../figures'

def plot_2D_scatter(X_train, X_test, Y_train, Y_test, classnames, label1,
                    label2, filename, xjitter=None, yjitter=None):

    classes = np.unique(Y_train)
    n_class = len(classes)

    plt.figure()
    fig, axes = plt.subplots(1, 2)
    fig.set_size_inches(12, 6)

    # training data
    plt.sca(axes[0])
    for i in xrange(n_class):
        xs = X_train[:, 0][Y_train==i]
        ys = X_train[:, 1][Y_train==i]
        if xjitter is not None:
            xs += np.random.randn(len(xs))*xjitter
        if yjitter is not None:
            ys += np.random.randn(len(ys))*yjitter
        plt.scatter(xs, ys, c=colors[i])
    plt.legend(classnames, bbox_to_anchor=(1.05, 1.05))  
    plt.xlabel(label1)
    plt.ylabel(label2)
    plt.title('Training data')

    # test data
    plt.sca(axes[1])
    for i in xrange(n_class):
        xs = X_test[:, 0][Y_test==i]
        ys = X_test[:, 1][Y_test==i]
        if xjitter is not None:
            xs += np.random.randn(len(xs))*xjitter
        if yjitter is not None:
            ys += np.random.randn(len(ys))*yjitter
        plt.scatter(xs, ys, c=colors[i])
    plt.legend(classnames, bbox_to_anchor=(1.05, 1.05))  
    plt.xlabel(label1)
    plt.ylabel(label2)
    plt.title('Test data')

    plt.savefig(join(figdir, filename))
    plt.close()

def plot_confusion_matrix(cm, labels, title, filename):
    print labels
    sums = np.sum(cm, axis=1)
    print cm, sums
    cm_norm = np.empty(cm.shape, dtype=float)
    for i in xrange(len(sums)):
        cm_norm[i, :] = cm[i, :]/np.float(sums[i])*100.
    print cm_norm

    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.matshow(cm_norm, cmap=plt.cm.gray_r)
    plt.title(title)
    plt.colorbar()
    ax.set_xticklabels([''] + labels)
    ax.set_yticklabels([''] + labels)
    plt.xlabel('Predicted')
    plt.ylabel('True')

    plt.savefig(join(figdir, filename))
    plt.close()

