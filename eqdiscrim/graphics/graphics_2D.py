import numpy as np
import matplotlib.pyplot as plt

colors = ['blue', 'lime', 'red', 'yellow', 'cyan', 'orange', 'black',
          'white', 'purple', 'green', 'grey']

# plot 2D clusters
def plot_2D_cluster_scatter(X, labels, at_name, filename):
    """
    Plots 2D clusters.
    X = (n,2) matrix of n instances with 2 attributes
    labels = cluster labels (n values, one per instance)
    at_names = attribute names (list of 2 strings)
    filename = filename for figure output

    """
    plt.figure()
    clust_names = np.unique(labels)
    nclust = len(clust_names)
    for i in xrange(nclust):
        cname = clust_names[i]
        x = X[:, 0][labels == cname]
        y = X[:, 1][labels == cname]
        plt.scatter(x, y, c=colors[i])
    plt.legend(clust_names, bbox_to_anchor=(1.05, 1.05))
    plt.xlabel(at_name[0])
    plt.ylabel(at_name[1])
    plt.savefig(filename)

def plot_att_hist_by_label(X_d, X_auth, att_range, nbins, att_name, filename):
    """
    Plots attribute histograms by author
    """

    # set up plot
    plt.figure()
    fig, axes = plt.subplots(1, 2)
    fig.set_size_inches(10, 5)

    # first plot - all together
    plt.sca(axes[0])
    n, bins, patches = plt.hist(X_d, nbins, range=att_range, normed=1,
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
        n, bins, patches = plt.hist(d, nbins, range=att_range, normed=1,
                                    histtype='step', color=colors[i])
    plt.legend(u_auth, bbox_to_anchor=(1.3, 1.05))
    plt.xlabel(att_name)
    plt.ylabel('Probability density')

    plt.savefig(filename)
