import numpy as np
from os.path import join
from preproc import GutenbergRichter
import matplotlib.pyplot as plt

colors = ['blue', 'lime', 'red', 'yellow', 'cyan', 'orange',
          'grey', 'purple', 'green', 'white', 'magenta']

epochs = [1970, 1978, 1990]

figdir = './figures'


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
        plt.scatter(x, y, c=colors[i], label=cname)
    plt.legend(bbox_to_anchor=(1.05, 1.05))
    # plt.legend(loc='upper left')
    plt.xlabel(at_name[0])
    plt.ylabel(at_name[1])
    plt.savefig(join(figdir, filename))
    plt.close()


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
            if len(x) > 0:
                plt.scatter(x, y, c=colors[i], label=cname)
        plt.legend(bbox_to_anchor=(1.05, 1.05))
        # plt.legend(loc='upper left')
        plt.xlabel(at_name[0])
        plt.ylabel(at_name[1])

    plt.savefig(join(figdir, filename))
    plt.close()


def plot_att_hist_by_label(X_d, labels, att_range, nbins, att_name, filename,
    hline=None):
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
    if hline is not None:
        plt.axhline(hline, lw=2, color='k')

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
    if hline is not None:
        plt.axhline(hline, lw=2, color='k')

    plt.savefig(join(figdir, filename))
    plt.close()


def plot_att_hist_by_label_deconv(X_d1, X_d2, labels1, labels2, att_range,
                                  nbins, att_name, filename, hline=None):
    """
    Plots attribute histograms by label (dividing X_d1 by X_d2).
    """

    # set up plot
    plt.figure()
    fig, axes = plt.subplots(1, 2)
    fig.set_size_inches(10, 5)

    # first plot - all together
    plt.sca(axes[0])
    pdf1, bins = np.histogram(X_d1, nbins, range=att_range, density=1)
    pdf2, bins = np.histogram(X_d2, nbins, range=att_range, density=1)
    if hline is not None:
        pdf = pdf1/pdf2*hline
    else:
        pdf = pdf1/pdf2
    width = bins[1]-bins[0]
    center = (bins[:-1]+bins[1:]) / 2.
    plt.bar(center, pdf, align='center', width=width, fc='none', ec='k')
    plt.xlabel(att_name)
    plt.ylabel('Probability density')
    if hline is not None:
        plt.axhline(hline, lw=2, color='k')

    # get unique authors
    u_auth = np.unique(labels1)
    n_auth = len(u_auth)

    # plot by author
    plt.sca(axes[1])
    for i in xrange(n_auth):
        author = u_auth[i]
        d1 = X_d1[:][labels1 == author]
        d2 = X_d2[:][labels2 == author]
        pdf1, bins = np.histogram(d1, nbins, range=att_range, density=1)
        pdf2, bins = np.histogram(d2, nbins, range=att_range, density=1)
        if hline is not None:
            pdf = pdf1/pdf2*hline
        else:
            pdf = pdf1/pdf2
        plt.bar(center, pdf, align='center', width=width, fc='none', ec=colors[i])
    plt.legend(u_auth, bbox_to_anchor=(1.3, 1.05))
    plt.xlabel(att_name)
    plt.ylabel('Probability density')
    if hline is not None:
        plt.axhline(hline, lw=2, color='k')

    plt.savefig(join(figdir, filename))
    plt.close()



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
    plt.close()


def plot_pie_comparison(labels1, values1, title1, labels2, values2, title2,
    filename):
    """
    Plots two pie charts in comparison
    """

    explode1 = np.ones(len(values1))*0.1
    explode2 = np.ones(len(values2))*0.1
    fig, axes = plt.subplots(1, 2)
    fig.set_size_inches(12, 5)
    plt.sca(axes[0])
    plt.pie(values1, labels=labels1, autopct='%.1f%%', explode=explode1,
        colors=colors)
    plt.title(title1)
    plt.sca(axes[1])
    plt.pie(values2, labels=labels2, autopct='%.1f%%', explode=explode2,
        colors=colors)
    plt.title(title2)

    plt.savefig(join(figdir, filename))
    plt.close()


def plot_bar_stacked(labels, values1, values2, label1, label2, ylabel, title,
                     filename, hline=None):
    """
    Plots a bar graph with values1 and values2 stacked.
    """

    N = len(labels)
    ind = np.arange(N)  # the x locations for the bars
    width = 0.55

    fig, axes = plt.subplots(1, 2)
    fig.set_size_inches(12, 5)
    plt.sca(axes[0])

    # plot values
    p1 = plt.bar(ind, values1, width, color='blue')
    p2 = plt.bar(ind, values2, width, color='red', bottom=values1)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.xticks(ind+width/2., labels)
    plt.legend((p1[0], p2[0]), (label1, label2))

    # plot fraction
    plt.sca(axes[1])
    f1 = values1/(values1+values2)
    f2 = values2/(values1+values2)
    p1 = plt.bar(ind, f1, width, color='blue')
    p2 = plt.bar(ind, f2, width, color='red', bottom=f1)
    plt.ylabel('Fraction')
    plt.title(title)
    plt.xticks(ind+width/2., labels)
    if hline is not None:
        plt.axhline(hline, color='k', lw=2)
    #plt.legend((p1[0], p2[0]), (label1, label2))

    plt.savefig(join(figdir, filename))
    plt.close()
