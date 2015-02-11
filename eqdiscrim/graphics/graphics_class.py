import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from os.path import join
from sklearn import metrics

colors = ['red', 'blue', 'green', 'cyan']
figdir = '../figures'

def evaluate_and_plot_2D(clf, X_train, X_test, Y_train, Y_test, classnames,
                         labels, title, filename, xjitter=None, yjitter=None):

    # do train and predict
    clf.fit(X_train, Y_train)
    Y_pred = clf.predict(X_test)
    c_matrix = metrics.confusion_matrix(Y_test, Y_pred)
    plot_confusion_matrix(c_matrix, classnames, title, "cm_%s"%filename)

    # get limits for plot
    X = np.vstack([X_train, X_test])
    xmin, xmax = X[:, 0].min() - .5, X[:, 0].max() + .5
    ymin, ymax = X[:, 1].min() - .5, X[:, 1].max() + .5

    step = 0.02
    xx, yy = np.meshgrid(np.arange(xmin, xmax, step),
                         np.arange(ymin, ymax, step))
    cm = plt.cm.RdBu
    cm_bright = ListedColormap(['#FF0000', '#0000FF'])


    # do plot
    fig = plt.figure()
    ax = fig.add_subplot(111)

    # plot the result
    try:
        Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
    except AttributeError: 
        Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
    Z = Z.reshape(xx.shape)
    ax.contourf(xx, yy, Z, cmap=cm, alpha=.8)

    # add training and test points
    xs, ys = X_train[:, 0], X_train[:, 1]
    if xjitter is not None:
        xs += np.random.randn(len(xs)) * xjitter
    if yjitter is not None:
        ys += np.random.randn(len(ys)) * yjitter
    ax.scatter(xs, ys, c=Y_train, cmap=cm_bright)

    xs, ys = X_test[:, 0], X_test[:, 1]
    if xjitter is not None:
        xs += np.random.randn(len(xs)) * xjitter
    if yjitter is not None:
        ys += np.random.randn(len(ys)) * yjitter
    ax.scatter(xs, ys, c=Y_test, cmap=cm_bright, alpha=0.6)

    plt.xlabel(labels[0])
    plt.ylabel(labels[1])
    plt.title(title)
    plt.savefig(join(figdir, filename))
    plt.close()


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



