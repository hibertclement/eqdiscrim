import numpy as np
import matplotlib.pyplot as plt
from pickle import load, dump
from cat_io.sihex_io import read_sihex_xls, read_notecto_lst
from cat_io.renass_io import read_renass, read_stations_fr
from preproc import latlon_to_xy, dist_to_n_closest_stations
from sklearn.preprocessing import StandardScaler
from sklearn import cluster
from graphics.graphics_2D import plot_2D_cluster_scatter
from graphics.graphics_2D import plot_2D_cluster_scatter_by_epoch
from graphics.graphics_2D import plot_att_hist_by_label
from dateutil import tz


to_zone = tz.gettz('Europe/Paris')
clust_file = 'clusteriser.dat'

do_clust = False

# read catalogs
#S_latlon, names_sta = read_renass()
S_latlon, names_sta = read_stations_fr()
X_latlon, y, names_latlon = read_sihex_xls(inout=False)
B_latlon, y, names_latlon = read_notecto_lst()

# extract the x and y coordinates
# for events
ilat = 2
ilon = 3
X, names_X = latlon_to_xy(X_latlon, names_latlon, ilat, ilon)
B, names_B = latlon_to_xy(B_latlon, names_latlon, ilat, ilon)
X_xy = X[:, [ilon, ilat]]
B_xy = B[:, [ilon, ilat]]

# extract the depth
idepth = 4
X_d = X[:, idepth]
B_d = B[:, idepth]

# extract the magnitude
imag = 5
X_m = X[:, imag]
B_m = B[:, imag]

# extract local time
i_time = 1
X_loctime = np.array([t.astimezone(to_zone) for t in X[:, i_time]])
X_hour = np.array([t.hour for t in X_loctime])
B_loctime = np.array([t.astimezone(to_zone) for t in B[:, i_time]])
B_hour = np.array([t.hour for t in B_loctime])

# save UTC origin time
X_otime = X[:, i_time]
B_otime = B[:, i_time]

# train the scaler on the events
scaler = StandardScaler().fit(X_xy)
X_xy = scaler.transform(X_xy)
B_xy = scaler.transform(B_xy)

# for stations
ilat = 1
ilon = 2
S, names_S = latlon_to_xy(S_latlon, names_sta, ilat, ilon)
S_xy = S[:, [ilon, ilat]]
S_xy = scaler.transform(S_xy)

# save UTC start and end times of stations
istart = 4
iend = 5
S_times = S[:, istart:iend+1]


# plot for sanity
plt.figure()
plt.scatter(X_xy[:, 0], X_xy[:,1], color='blue', label='ke')
plt.scatter(B_xy[:, 0], B_xy[:,1], color='yellow', label='km/sm')
plt.scatter(S_xy[:, 0], S_xy[:,1], marker='v', color='red', label='station')
plt.xlabel('Reduced x coordinate')
plt.ylabel('Reduced y coordinate')
plt.legend()
plt.savefig('../figures/sihex_events_stations.png')

if do_clust:

    # get distance from 3rd station (using timing info)
    nev, nd = X_xy.shape
    X_xyt = np.hstack((X_xy, X_otime.reshape(nev, 1)))
    S_xyt = np.hstack((S_xy, S_times))
    d3sta = dist_to_n_closest_stations(X_xyt, S_xyt, 3, timing=True)

    # create a cluster-izer on this distance
    nclust = 3
    clf = cluster.KMeans(init='k-means++', n_clusters=nclust, random_state=42)
    clf.fit(d3sta)

    # dump clusterer to file
    f_ = open(clust_file,'w')
    dump(clf, f_)
    f_.close()

# read clusterer from file
f_ = open(clust_file, 'r')
clf = load(f_)
f_.close()

# use it to predict labels for non-tectonic events
# get distance from 3rd station (using timing info)
nev, nd = B_xy.shape
B_xyt = np.hstack((B_xy, B_otime.reshape(nev, 1)))
S_xyt = np.hstack((S_xy, S_times))
B_d3sta = dist_to_n_closest_stations(B_xyt, S_xyt, 3, timing=True)
# do prediction
B_labels = clf.predict(B_d3sta)

# plot geographic clusters
plot_2D_cluster_scatter(X_xy, clf.labels_,
                        ('Reduced x coordinate', 'Reduced y coordinate'),
                        'clusters_dist_to_3_closest_stations.png')
plot_2D_cluster_scatter(B_xy, B_labels,
                        ('Reduced x coordinate', 'Reduced y coordinate'),
                        'notecto_clusters_dist_to_3_closest_stations.png')

# plot geographic clusters by epoch
plot_2D_cluster_scatter_by_epoch(X_xy, X_otime, clf.labels_,
                        ('Reduced x coordinate', 'Reduced y coordinate'),
                        'clusters_dist_to_3_closest_stations_by_epoch.png')
plot_2D_cluster_scatter_by_epoch(B_xy, B_otime, B_labels,
                        ('Reduced x coordinate', 'Reduced y coordinate'),
                        'notecto_clusters_dist_to_3_closest_stations_by_epoch.png')

# plot magnitude as a function of clusters
nbins = 20
mag_range=(0, 6)
plot_att_hist_by_label(X_m, clf.labels_, mag_range, nbins, 'Mw',
                       'mag_pdf_by_station_cluster.png')
plot_att_hist_by_label(B_m, B_labels, mag_range, nbins, 'Mw',
                       'notecto_mag_pdf_by_station_cluster.png')

# plot local hour as a function of clusters
nbins = 24
time_range = (0, 23)
plot_att_hist_by_label(X_hour, clf.labels_, time_range, nbins, 'Local hour',
                       'hour_pdf_by_station_cluster.png')
plot_att_hist_by_label(B_hour, B_labels, time_range, nbins, 'Local hour',
                       'notecto_hour_pdf_by_station_cluster.png')
