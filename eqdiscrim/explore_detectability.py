import numpy as np
import matplotlib.pyplot as plt
from pickle import load, dump
from sklearn import cluster
from cat_io.sihex_io import read_sihex_tidy
from graphics.graphics_2D import plot_2D_cluster_scatter
from graphics.graphics_2D import plot_2D_cluster_scatter_by_epoch
from graphics.graphics_2D import plot_att_hist_by_label
from graphics.graphics_2D import plot_GR_by_label


Xtable = '../static_catalogs/sihex_tidy_earthquakes.dat'
Btable = '../static_catalogs/sihex_tidy_blasts.dat'
Atable = '../static_catalogs/sihex_tidy_all.dat'
Hfile = '../static_catalogs/sihex_tidy_header.txt'

Stable = '../static_catalogs/sihex_tidy_stations.dat'
SHfile = '../static_catalogs/sihex_tidy_stations_header.txt'

clust_file = 'clusteriser.dat'
do_clust = False
nclust = 3

# read catalogs separatelly
X, coldict = read_sihex_tidy(Xtable, Hfile)
B, coldict = read_sihex_tidy(Btable, Hfile)
S, scoldict = read_sihex_tidy(Stable, SHfile)

# extract columns of interest
ix = coldict['X']
iy = coldict['Y']
i1 = coldict['DistanceStation1']
i2 = coldict['DistanceStation2']
i3 = coldict['DistanceStation3']
im = coldict['Mw']
itime = coldict['OriginTime']
ih = coldict['LocalHour']
iw = coldict['LocalWeekday']

isx = scoldict['X']
isy = scoldict['Y']

# extract x and y info
X_xy = X[:, [ix, iy]]
B_xy = B[:, [ix, iy]]
S_xy = S[:, [isx, isy]]

# plot for sanity
plt.figure()
plt.scatter(X_xy[:, 0], X_xy[:, 1], color='blue', label='ke')
plt.scatter(B_xy[:, 0], B_xy[:, 1], color='yellow', label='km/sm')
plt.scatter(S_xy[:, 0], S_xy[:, 1], marker='v', color='red', label='station')
plt.xlabel('Reduced x coordinate')
plt.ylabel('Reduced y coordinate')
plt.legend()
plt.savefig('../figures/sihex_events_stations.png')

# distance from station
 
if do_clust:

   # get distance from 3rd station (using timing info)
    X_d3sta = X[:, [i1, i2, i3]]

    # create a cluster-izer on this distance
    clf = cluster.KMeans(init='k-means++', n_clusters=nclust, random_state=42)
    clf.fit(X_d3sta)

    # dump clusterer to file
    f_ = open(clust_file, 'w')
    dump(clf, f_)
    f_.close()

# read clusterer from file
f_ = open(clust_file, 'r')
clf = load(f_)
f_.close()

# use it to predict labels for non-tectonic events
# get distance from 3rd station (using timing info)
B_d3sta = B[:, [i1, i2, i3]]
B_labels = clf.predict(B_d3sta)

# plot geographic clusters
plot_2D_cluster_scatter(X_xy, clf.labels_,
                        ('Reduced x coordinate', 'Reduced y coordinate'),
                        'clusters_dist_to_3_closest_stations.png')
plot_2D_cluster_scatter(B_xy, B_labels,
                        ('Reduced x coordinate', 'Reduced y coordinate'),
                        'notecto_clusters_dist_to_3_closest_stations.png')

X_otime = X[:, itime]
B_otime = B[:, itime]
# plot geographic clusters by epoch
plot_2D_cluster_scatter_by_epoch(
    X_xy, X_otime, clf.labels_,
    ('Reduced x coordinate', 'Reduced y coordinate'),
    'clusters_dist_to_3_closest_stations_by_epoch.png')
plot_2D_cluster_scatter_by_epoch(
    B_xy, B_otime, B_labels,
    ('Reduced x coordinate', 'Reduced y coordinate'),
    'notecto_clusters_dist_to_3_closest_stations_by_epoch.png')

# plot magnitude as a function of clusters
X_m = X[:, im]
B_m = B[:, im]
nbins = 20
mag_range = (0, 6)
plot_att_hist_by_label(X_m, clf.labels_, mag_range, nbins, 'Mw',
                       'mag_pdf_by_station_cluster.png')
plot_att_hist_by_label(B_m, B_labels, mag_range, nbins, 'Mw',
                       'notecto_mag_pdf_by_station_cluster.png')

# plot local hour as a function of clusters
X_hour = X[:, ih]
B_hour = B[:, ih]
nbins = 24
time_range = (0, 23)
plot_att_hist_by_label(X_hour, clf.labels_, time_range, nbins, 'Local hour',
                       'hour_pdf_by_station_cluster.png')
plot_att_hist_by_label(B_hour, B_labels, time_range, nbins, 'Local hour',
                       'notecto_hour_pdf_by_station_cluster.png')

# plot local weekday as a function of clusters
X_wd = X[:, iw]
B_wd = B[:, iw]
nbins = 7
time_range = (1, 7)
plot_att_hist_by_label(X_wd, clf.labels_, time_range, nbins, 'Local weekday',
                       'weekday_pdf_by_station_cluster.png')
plot_att_hist_by_label(B_wd, B_labels, time_range, nbins, 'Local weekday',
                       'notecto_weekday_pdf_by_station_cluster.png')

# plot GR
min_mag = np.min(X_m)
max_mag = np.max(X_m)
mag_step = 0.1
plot_GR_by_label(X_m, clf.labels_, min_mag, max_mag, mag_step,
                 'GR_by_station_cluster.png')
