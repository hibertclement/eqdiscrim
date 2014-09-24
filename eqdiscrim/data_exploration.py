import numpy as np
#import matplotlib.pyplot as plt
from cat_io.sihex_io import read_sihex_xls
from preproc import latlon_to_xy
from sklearn.preprocessing import StandardScaler
from sklearn import cluster
from graphics.graphics_2D import plot_2D_cluster_scatter
from graphics.graphics_2D import plot_att_hist_by_label
from dateutil import tz


to_zone = tz.gettz('Europe/Paris')


# read catalog
Xlatlon, y, names_latlon = read_sihex_xls()

nev, natt = Xlatlon.shape

# extract the x and y coordinates
print Xlatlon.shape, names_latlon
ilat = 2
ilon = 3
X, names = latlon_to_xy(Xlatlon, names_latlon, ilat, ilon)
X_xy = X[:, [ilon, ilat]]

# extract the depth
idepth = 4
X_d = X[:, idepth]

# extract the magnitude
imag = 5
X_m = X[:, imag]

# extract authors
iauth = 6
X_auth = X[:, iauth]

# extract local time
i_time = 1
X_loctime = np.array([t.astimezone(to_zone) for t in X[:, i_time]])


# scale geographical coordinates
scaler = StandardScaler().fit(X_xy)
X_xy = scaler.transform(X_xy)
X_year = np.array([t.year for t in X_loctime])
X_hour = np.array([t.hour for t in X_loctime])


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

# plot depth and magnitude as a function of author
nbins = 20
depth_range=(-3, 40)
plot_att_hist_by_label(X_d, X_auth, depth_range, nbins, 'Depth (km)',
                       'depth_pdf_by_author.png')
mag_range=(0, 6)
plot_att_hist_by_label(X_m, X_auth, mag_range, nbins, 'Mw',
                       'mag_pdf_by_author.png')

# plot depth and magnitude as a function of cluster
depth_range=(-3, 40)
plot_att_hist_by_label(X_d, clf.labels_, depth_range, nbins, 'Depth (km)',
                       'depth_pdf_by_cluster.png')
mag_range=(0, 6)
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


