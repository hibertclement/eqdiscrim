import numpy as np
import matplotlib.pyplot as plt

colors = ['blue', 'lime', 'red', 'yellow', 'cyan', 'purple', 'black',
          'white', 'orange', 'green', 'grey']

# plot 2D clusters
def plot_2D_cluster_scatter(X, labels, at_name, filename):
    """
    Plots 2D clusters.
    X = (n,2) matrix of n instances with 2 attributes
    labels = cluster labels (n values, one per instance)
    at_names = attribute names (list of 2 strings)
    filename = filename for figure output

    """
    nclust = len(np.unique(labels))
    for i in xrange(nclust):
        x = X[:, 0][labels == i]
        y = X[:, 1][labels == i]
        plt.scatter(x, y, c=colors[i])
    plt.legend(np.arange(nclust))
    plt.xlabel(at_name[0])
    plt.ylabel(at_name[1])
    plt.savefig(filename)

def plot_att_hist_by_author(X_d, X_auth, att_range, att_name, filename):
    """
    Plots attribute histograms by author
    """

    # set up plot
    plt.figure()
    fig, axes = plt.subplots(1, 2)
    fig.set_size_inches(10, 5)

    # first plot - all together
    plt.sca(axes[0])
    n, bins, patches = plt.hist(X_d, 20, range=att_range, normed=1,
                                histtype='step', color='black')
    plt.xlabel(att_name)
    plt.ylabel('Probability density')

    # get unique authors
    u_auth = np.unique(X_auth)
    n_auth = len(u_auth)

    # plot by author
    plt.sca(axes[1])
    for i in xrange(n_auth):
        author = u_auth[i]
        d = X_d[:][X_auth == author]
        n, bins, patches = plt.hist(d, 20, range=att_range, normed=1,
                                    histtype='step', color=colors[i])
    plt.legend(u_auth)
    plt.xlabel(att_name)
    plt.ylabel('Probability density')

    plt.savefig(filename)
