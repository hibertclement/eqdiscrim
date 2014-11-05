"""
Script to run all the data processing steps to get from raw to tidy data
"""
import numpy as np
from dateutil import tz
from pickle import dump
from sklearn.preprocessing import StandardScaler
from cat_io.sihex_io import read_sihex_xls, read_notecto_lst
from cat_io.renass_io import read_stations_fr
from preproc import latlon_to_xy, dist_to_n_closest_stations


to_zone = tz.gettz('Europe/Paris')
N_close = 3  # c-closest stations (used for clustering)

Xtable = '../static_catalogs/sihex_tidy_earthquakes.dat'
Btable = '../static_catalogs/sihex_tidy_blasts.dat'
Atable = '../static_catalogs/sihex_tidy_all.dat'
Hfile = '../static_catalogs/sihex_tidy_header.txt'

Stable = '../static_catalogs/sihex_tidy_stations.dat'
SHfile = '../static_catalogs/sihex_tidy_stations_header.txt'


# #############
# read catalogs
# #############
print 'Reading and parsing catalogs...'
S_latlon, names_sta = read_stations_fr()
X_latlon, X_y, X_names_latlon = read_sihex_xls(inout=False)
B_latlon, B_y, B_names_latlon = read_notecto_lst()

# #####################
# extract the event IDs
# #####################

print 'Extracting event ids...'
iid = 0
X_id = X_latlon[:, iid]
B_id = B_latlon[:, iid]
S_id = S_latlon[:, iid]

# #############################################################################
# extract the lat-lon coordinates (and transform them to scaled xy coordinates)
# #############################################################################

# get event coordinates
print 'Extracting and transforming coordinates...'
ilat = 2
ilon = 3
X, X_names = latlon_to_xy(X_latlon, X_names_latlon, ilat, ilon)
B, B_names = latlon_to_xy(B_latlon, B_names_latlon, ilat, ilon)
X_xy = X[:, [ilon, ilat]]
B_xy = B[:, [ilon, ilat]]

# train the scaler on the events
scaler = StandardScaler().fit(X_xy)

# scale the event coordinates
X_xy = scaler.transform(X_xy)
B_xy = scaler.transform(B_xy)

# get station coordinates
ilat = 1
ilon = 2
S, S_names = latlon_to_xy(S_latlon, names_sta, ilat, ilon)
S_xy = S[:, [ilon, ilat]]

# scale the station coordinates
S_xy = scaler.transform(S_xy)

# ######################
# extract the magnitudes
# ######################

print 'Extracting magnitudes...'
imag = 5
X_m = X[:, imag]
B_m = B[:, imag]

# ###########################
# extract the UTC origin time
# ###########################

print 'Extracting origin times...'
itime = 1
X_otime = X[:, itime]
B_otime = B[:, itime]

# ####################################
# obtain local origin time and weekday
# ####################################

print 'Extracting local hour and weekday...'
X_loctime = np.array([t.astimezone(to_zone) for t in X[:, itime]])
X_hour = np.array([t.hour for t in X_loctime])
X_weekday = np.array([t.isoweekday() for t in X_loctime])
B_loctime = np.array([t.astimezone(to_zone) for t in B[:, itime]])
B_hour = np.array([t.hour for t in B_loctime])
B_weekday = np.array([t.isoweekday() for t in B_loctime])

# ###########################################
# obtain distance to N_close closest stations
# ###########################################

print 'Extracting distance to %d closest stations...' % N_close
# get start and end times of stations
istart = 4
iend = 5
S_times = S[:, istart:iend+1]
S_xyt = np.hstack((S_xy, S_times))

# get distances to N_close closest stations (events)
nev, nd = X_xy.shape
X_xyt = np.hstack((X_xy, X_otime.reshape(nev, 1)))
X_d3sta = dist_to_n_closest_stations(X_xyt, S_xyt, N_close, timing=True)

# get distances to N_close closest stations (blasts)
nev, nd = B_xy.shape
B_xyt = np.hstack((B_xy, B_otime.reshape(nev, 1)))
B_d3sta = dist_to_n_closest_stations(B_xyt, S_xyt, N_close, timing=True)

# ###################################################
# combining information into a single matrix per type
# ###################################################

header = "ID OriginTime X Y Mw LocalHour LocalWeekday DistanceStation1 DistanceStation2 DistanceStation3 Type"
sta_header = "Name X Y StartTime EndTime"

print 'Combining into tidy tables...'
nev, nd = X_xy.shape
X = np.hstack((X_id.reshape(nev, 1), X_otime.reshape(nev, 1), X_xy,
               X_m.reshape(nev, 1), X_hour.reshape(nev, 1),
               X_weekday.reshape(nev, 1), X_d3sta, X_y.reshape(nev, 1)))

nev, nd = B_xy.shape
B = np.hstack((B_id.reshape(nev, 1), B_otime.reshape(nev, 1), B_xy,
               B_m.reshape(nev, 1), B_hour.reshape(nev, 1),
               B_weekday.reshape(nev, 1), B_d3sta, B_y.reshape(nev, 1)))

A = np.vstack((X, B))

nst, nd = S_xy.shape
S = np.hstack((S_id.reshape(nst, 1), S_xyt))

# ###################
# writing tidy tables
# ###################

print 'Writing tidy tables...'
print Xtable
f_ = open(Xtable, 'w')
dump(X, f_)
f_.close()

print Btable
f_ = open(Btable, 'w')
dump(B, f_)
f_.close()

print Atable
f_ = open(Atable, 'w')
dump(A, f_)
f_.close()

print Hfile
f_ = open(Hfile, 'w')
f_.write(header)
f_.close()

print Stable
f_ = open(Stable, 'w')
dump(S, f_)
f_.close()

print SHfile
f_ = open(SHfile, 'w')
f_.write(sta_header)
f_.close()
