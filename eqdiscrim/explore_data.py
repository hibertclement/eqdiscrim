import numpy as np
from cat_io.sihex_io import read_sihex_tidy
from sklearn import cluster
from graphics.graphics_2D import plot_2D_cluster_scatter
from graphics.graphics_2D import plot_att_hist_by_label
from dateutil import tz

Xtable = '../static_catalogs/sihex_tidy_earthquakes.dat'
Hfile = '../static_catalogs/sihex_tidy_header.txt'

# read catalog
X, coldict = read_sihex_tidy(Xtable, Hfile)

# get indexes
ix = coldict['X']
iy = coldict['Y']
ia = coldict['Author']
im = coldict['Mw']
ih = coldict['LocalHour']
it = coldict['OriginTime']

# extract info
X_xy = X[:, [ix, iy]]
X_auth = X[:, ia]
X_m = X[:, im]
X_hour = X[:, ih]
X_otime = X[:, it]
nev = len(X_otime)
X_year = np.array([otime.year for otime in X_otime])


# create a cluster-izer
nclust = 5
clf = cluster.KMeans(init='k-means++', n_clusters=nclust, random_state=42)
clf.fit(X_xy)

# plot geographic clusters
plot_2D_cluster_scatter(X_xy, clf.labels_,
                        ('Reduced x coordinate', 'Reduced y coorindate'),
                        'clusters_xy.png')

plot_2D_cluster_scatter(X_xy, X_auth,
                        ('Reduced x coordinate', 'Reduced y coorindate'),
                        'clusters_by_author.png')

# plot magnitude as a function of author
nbins = 20
mag_range = (0, 6)
plot_att_hist_by_label(X_m, X_auth, mag_range, nbins, 'Mw',
                       'mag_pdf_by_author.png')

# plot magnitude as a function of cluster
mag_range = (0, 6)
plot_att_hist_by_label(X_m, clf.labels_, mag_range, nbins, 'Mw',
                       'mag_pdf_by_cluster.png')

# plot year as a function of author and cluster
nbins = 24
time_range = (1962, 2010)
plot_att_hist_by_label(X_year, X_auth, time_range, nbins, 'Year',
                       'year_pdf_by_author.png')
plot_att_hist_by_label(X_year, clf.labels_, time_range, nbins, 'Year',
                       'year_pdf_by_cluster.png')

# plot hour as a function of author and cluster
nbins = 24
time_range = (0, 23)
plot_att_hist_by_label(X_hour, X_auth, time_range, nbins, 'Local hour',
                       'hour_pdf_by_author.png')
plot_att_hist_by_label(X_hour, clf.labels_, time_range, nbins, 'Local hour',
                       'hour_pdf_by_cluster.png')
