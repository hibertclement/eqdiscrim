import numpy as np
import matplotlib.pyplot as plt
from cat_io.sihex_io import read_sihex_xls
from cat_io.renass_io import read_renass
from preproc import latlon_to_xy, dist_to_n_closest_stations
from sklearn.preprocessing import StandardScaler
from sklearn import cluster
from graphics.graphics_2D import plot_2D_cluster_scatter
from graphics.graphics_2D import plot_att_hist_by_label
from dateutil import tz


to_zone = tz.gettz('Europe/Paris')

# read catalogs
S_latlon, names_sta = read_renass()
X_latlon, y, names_latlon = read_sihex_xls()

# extract the x and y coordinates
# for events
ilat = 2
ilon = 3
X, names_X = latlon_to_xy(X_latlon, names_latlon, ilat, ilon)
X_xy = X[:, [ilon, ilat]]

# extract the depth
idepth = 4
X_d = X[:, idepth]

# extract the magnitude
imag = 5
X_m = X[:, imag]

# extract local time
i_time = 1
X_loctime = np.array([t.astimezone(to_zone) for t in X[:, i_time]])
X_year = np.array([t.year for t in X_loctime])
X_hour = np.array([t.hour for t in X_loctime])

# train the scaler on the events
scaler = StandardScaler().fit(X_xy)
X_xy = scaler.transform(X_xy)

# for stations
ilat = 1
ilon = 2
S, names_S = latlon_to_xy(S_latlon, names_sta, ilat, ilon)
S_xy = S[:, [ilon, ilat]]
S_xy = scaler.transform(S_xy)


# plot for sanity
plt.figure()
plt.scatter(X_xy[:, 0], X_xy[:,1])
plt.scatter(S_xy[:, 0], S_xy[:,1], marker='v', color='red')
plt.xlabel('Reduced x coordinate')
plt.xlabel('Reduced y coordinate')
plt.savefig('sihex_events_stations.png')

# get distance from 3rd station
d3sta = dist_to_n_closest_stations(X_xy, S_xy, 3)

# create a cluster-izer on this distance
nclust = 5
clf = cluster.KMeans(init='k-means++', n_clusters=nclust, random_state=42)
clf.fit(d3sta)

# plot geographic clusters
plot_2D_cluster_scatter(X_xy, clf.labels_,
                        ('Reduced x coordinate', 'Reduced y coorindate'),
                        'clusters_dist_to_3_closest_stations.png')


# plot magnitude as a function of clusters
nbins = 20
mag_range=(0, 6)
plot_att_hist_by_label(X_m, clf.labels_, mag_range, nbins, 'Mw',
                       'mag_pdf_by_station_cluster.png')
# depth
depth_range=(-3, 40)
plot_att_hist_by_label(X_d, clf.labels_, depth_range, nbins, 'Depth (km)',
                       'depth_pdf_by_station_cluster.png')

# local hour
nbins = 24
time_range = (0, 23)
plot_att_hist_by_label(X_hour, clf.labels_, time_range, nbins, 'Local hour',
                       'hour_pdf_by_station_cluster.png')
