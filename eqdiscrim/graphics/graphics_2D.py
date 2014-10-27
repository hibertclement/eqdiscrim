import numpy as np
from os.path import join
from preproc import GutenbergRichter
import matplotlib.pyplot as plt

colors = ['blue', 'lime', 'red', 'yellow', 'cyan', 'orange', 'black',
          'white', 'purple', 'green', 'grey']

epochs = [1970, 1978, 1990]

figdir = '../figures'


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
    plt.savefig(join(figdir, filename))


def plot_2D_cluster_scatter_by_epoch(X, Xt, labels, at_name, filename):
    """
    Plots 2D clusters.
    X = (n,2) matrix of n instances with 2 attributes
    Xt = (n,1) vector of timing info
    labels = cluster labels (n values, one per instance)
    at_names = attribute names (list of 2 strings)
    filename = filename for figure output

    """
    # set up plot
    plt.figure()
    fig, axes = plt.subplots(2, 2)
    fig.set_size_inches(12, 12)

    clust_names = np.unique(labels)
    nclust = len(clust_names)
    years = np.array([Xt[i].year for i in xrange(len(Xt))])
    n_epochs = len(epochs)+1

    # loop over epochs
    for ie in xrange(n_epochs):
        plt.sca(axes[ie/2, np.mod(ie, 2)])
        for i in xrange(nclust):
            cname = clust_names[i]
            if ie == 0:
                x = X[:, 0][(labels == cname) & (years < epochs[ie])]
                y = X[:, 1][(labels == cname) & (years < epochs[ie])]
                plt.title('< %d' % epochs[ie])
            elif ie == n_epochs-1:
                x = X[:, 0][(labels == cname) & (years >= epochs[-1])]
                y = X[:, 1][(labels == cname) & (years >= epochs[-1])]
                plt.title('> %d' % epochs[-1])
            else:
                x = X[:, 0][(labels == cname) & (years >= epochs[ie-1]) &
                            (years < epochs[ie])]
                y = X[:, 1][(labels == cname) & (years >= epochs[ie-1]) &
                            (years < epochs[ie])]
                plt.title('%d - %d' % (epochs[ie-1], epochs[ie]))
            plt.scatter(x, y, c=colors[i])
        plt.legend(clust_names, bbox_to_anchor=(1.05, 1.05))
        plt.xlabel(at_name[0])
        plt.ylabel(at_name[1])

    plt.savefig(join(figdir, filename))


def plot_att_hist_by_label(X_d, labels, att_range, nbins, att_name, filename):
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
    u_auth = np.unique(labels)
    n_auth = len(u_auth)

    # plot by author
    plt.sca(axes[1])
    for i in xrange(n_auth):
        author = u_auth[i]
        d = X_d[:][labels == author]
        n, bins, patches = plt.hist(d, nbins, range=att_range, normed=1,
                                    histtype='step', color=colors[i])
    plt.legend(u_auth, bbox_to_anchor=(1.3, 1.05))
    plt.xlabel(att_name)
    plt.ylabel('Probability density')

    plt.savefig(join(figdir, filename))


def plot_GR_by_label(magnitudes, labels, min_mag, max_mag, mag_step, filename):
    """
    Plots GR plots by label.
    """

    # set up plot
    plt.figure()
    fig, axes = plt.subplots(1, 2)
    fig.set_size_inches(10, 5)

    # first plot - all together
    log10N, mags = GutenbergRichter(magnitudes, min_mag, max_mag, mag_step)
    plt.sca(axes[0])
    plt.scatter(mags[0:-1], log10N, c='black')
    plt.xlabel('Mw')
    plt.ylabel('log10(N)')
    plt.title('Gutenberg-Richter')

    # get unique labels
    u_labels = np.unique(labels)
    n_labels = len(u_labels)

    # do the second plot
    plt.sca(axes[1])
    for i in xrange(n_labels):
        m_sub = magnitudes[:][labels == u_labels[i]]
        log10N, mags = GutenbergRichter(m_sub, min_mag, max_mag, mag_step)
        plt.scatter(mags[0:-1], log10N, color=colors[i], label=u_labels[i])
    plt.legend(bbox_to_anchor=(1.05, 1.05))
    plt.xlabel('Mw')
    plt.ylabel('log10(N)')
    plt.title('Gutenberg-Richter')

    plt.savefig(join(figdir, filename))
