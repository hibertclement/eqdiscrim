import numpy as np
from pickle import load
from cat_io.sihex_io import read_sihex_tidy
from graphics.graphics_2D import plot_pie_comparison, plot_bar_stacked
from graphics.graphics_2D import plot_2D_cluster_scatter
from graphics.graphics_2D import plot_2D_cluster_scatter_by_epoch
from graphics.graphics_2D import plot_att_hist_by_label
from graphics.graphics_2D import plot_att_hist_by_label_deconv


# ## Exploring non-tectonic events that occur at night or at the weekends
# 
# Some non-tectonic events from the SiHex catalogue seem to occur at night and
# at weekends (see figures that were created using the
# **explore_detectability.py** script). The big question is why...

# ### Step 1 : extract night and weekend events
# 
# The night-time events and the weekend non-tectonic events can easily be
# extracted from the full catalogs. For now, let's just see what types they
# have...

clust_file = 'clusteriser.dat'

Xtable = '../static_catalogs/sihex_tidy_earthquakes.dat'
Btable = '../static_catalogs/sihex_tidy_blasts.dat'
Hfile = '../static_catalogs/sihex_tidy_header.txt'

print "Extracting night and weekend events..."
B, coldict = read_sihex_tidy(Btable, Hfile)
X, coldict = read_sihex_tidy(Xtable, Hfile)
B_hour = B[:, coldict['LocalHour']]
B_weekday = B[:, coldict['LocalWeekday']]
B_night = B[:, :][(B_hour < 8.0) | (B_hour > 19)]
B_weekend = B[:, :][B_weekday >= 6]

# get some simple statistics
nev, nd = B.shape
nev_night, nd = B_night.shape
nev_weekend, nd = B_weekend.shape

# lump night and weekend events together for the following anaysis
nw_ids = np.union1d(B_night[:, coldict['ID']], B_weekend[:, coldict['ID']])
n_nw = len(nw_ids)
print\
"%d (i.e. %.2f%%) non-tectonic events occur during the night or at weekends."\
    % (n_nw, n_nw/np.float(nev)*100)

# get the night and weekend event from ids
B_night_weekend = np.empty((n_nw, nd), dtype=object)
for i in xrange(n_nw):
    B_night_weekend[i, :] = B[:, :][B[:, coldict['ID']] == nw_ids[i]]

print "Computing the frequencies..."
# count the type frequencies for all the non-tectonics and also only those
# that occur at night and at weekends
all_types = np.unique(B[:, coldict['Type']])
n_types = {}
n_types_night_weekend = {}
n_types_workday = {}
for t in all_types:
    n_types_night_weekend[t] =\
        np.sum([B_night_weekend[:, coldict['Type']] == t])
    n_types[t] = np.sum([B[:, coldict['Type']] == t])
    n_types_workday[t] = n_types[t] - n_types_night_weekend[t]

# print information 
print 'Types', n_types.keys()
print 'All', n_types.values()
print 'Workday', n_types_workday.values()
print 'Night or weekend', n_types_night_weekend.values()

keys = ["km", "sm", "me", "kr", "sr", "ki", "si", "uk"]
nk = len(keys)
workday = np.empty(nk, dtype=np.float)
night_weekend = np.empty(nk, dtype=np.float)
for i in xrange(nk):
    workday[i] = n_types_workday[keys[i]]
    night_weekend[i] = n_types_night_weekend[keys[i]]

print "Plotting the frequencies..."
plot_pie_comparison(keys, workday, 'Workday (%d)' % (nev-n_nw),
                    keys, night_weekend, 'Night or weekend (%d)' %
                    n_nw, 'notecto_type_pie_comparison.png')

plot_bar_stacked(keys, workday, night_weekend, 'Workday', 'Night or weekend',
                 'Number of events', 'Non-tectonic event types',
                 'notecto_type_bar_chart.png', hline=60/float(24*7))


print "Plotting the scatter plots..."
B_xy = B[:, [coldict['X'], coldict['Y']]]
plot_2D_cluster_scatter(B_xy, B[:, coldict['Type']],
                        ['Reduced x coordinate', 'Reduced y coordinate'],
                        'notecto_type_scatterplot.png')
plot_2D_cluster_scatter_by_epoch(B_xy, B[:, coldict['OriginTime']],
                        B[:, coldict['Type']],
                        ['Reduced x coordinate', 'Reduced y coordinate'],
                        'notecto_type_scatterplot_by_epoch.png')
B_km = B[:, :][B[:, coldict['Type']] == 'km']
B_sm = B[:, :][B[:, coldict['Type']] == 'sm']
B_me = B[:, :][B[:, coldict['Type']] == 'me']
B_kr = B[:, :][B[:, coldict['Type']] == 'kr']
B_sr = B[:, :][B[:, coldict['Type']] == 'sr']
B_ki = B[:, :][B[:, coldict['Type']] == 'ki']
B_si = B[:, :][B[:, coldict['Type']] == 'si']
B_uk = B[:, :][B[:, coldict['Type']] == 'uk']

B_ksm = np.vstack((B_km, B_sm, B_me))
B_ksri = np.vstack((B_kr, B_sr, B_ki, B_si))

B_ant = np.vstack((B_km, B_sm, B_me))
B_uni = np.vstack((B_ki, B_si, B_uk))
B_rok = np.vstack((B_kr, B_sr))

# read clusterer
f_ = open(clust_file, 'r')
clf = load(f_)
f_.close()

# use it to predict labels for non-tectonic events
# get distance from 3rd station (using timing info)
i1 = coldict['DistanceStation1']
i2 = coldict['DistanceStation2']
i3 = coldict['DistanceStation3']
B_d3sta_ant = B_ant[:, [i1, i2, i3]]
B_d3sta_uni = B_uni[:, [i1, i2, i3]]
B_d3sta_rok = B_rok[:, [i1, i2, i3]]
B_labels_ant = clf.predict(B_d3sta_ant)
B_labels_uni = clf.predict(B_d3sta_uni)
B_labels_rok = clf.predict(B_d3sta_rok)

plot_2D_cluster_scatter(B_ksm[:, [coldict['X'], coldict['Y']]],
                        B_ksm[:, coldict['Type']],
                        ['Reduced x coordinate', 'Reduced y coordinate'],
                        'notecto_kmsmme_scatterplot.png')
plot_2D_cluster_scatter_by_epoch(B_ksm[:, [coldict['X'], coldict['Y']]],
                                 B_ksm[:, coldict['OriginTime']],
                                 B_ksm[:, coldict['Type']],
                                 ['Reduced x coordinate',
                                  'Reduced y coordinate'],
                                 'notecto_kmsmme_scatterplot_by_epoch.png')

plot_2D_cluster_scatter(B_ksri[:, [coldict['X'], coldict['Y']]],
                        B_ksri[:, coldict['Type']],
                        ['Reduced x coordinate', 'Reduced y coordinate'],
                        'notecto_krsrkisi_scatterplot.png')
plot_2D_cluster_scatter_by_epoch(B_ksri[:, [coldict['X'], coldict['Y']]],
                        B_ksri[:, coldict['OriginTime']],
                        B_ksri[:, coldict['Type']],
                        ['Reduced x coordinate', 'Reduced y coordinate'],
                        'notecto_krsrkisi_scatterplot_by_epoch.png')

print "Plotting the histograms as a function of type..."
B_hour = B[:, coldict['LocalHour']]
nbins = 24
time_range = (0, 23)
plot_att_hist_by_label(B_hour, B[:, coldict['Type']], time_range, nbins,
                       'Local hour', 'notecto_hour_pdf_by_type.png',
                       hline=1/24.)
plot_att_hist_by_label(B_ant[:, coldict['LocalHour']],
                       B_ant[:, coldict['Type']], time_range, nbins,
                       'Local hour', 'notecto_ant_hour_pdf_by_type.png',
                       hline=1/24.)
plot_att_hist_by_label(B_ant[:, coldict['LocalHour']],
                       B_labels_ant, time_range, nbins,
                       'Local hour', 'notecto_ant_hour_pdf_by_station_cluster.png',
                       hline=1/24.)
plot_att_hist_by_label(B_uni[:, coldict['LocalHour']],
                       B_uni[:, coldict['Type']], time_range, nbins,
                       'Local hour', 'notecto_uni_hour_pdf_by_type.png',
                       hline=1/24.)
plot_att_hist_by_label(B_uni[:, coldict['LocalHour']],
                       B_labels_uni, time_range, nbins,
                       'Local hour', 'notecto_uni_hour_pdf_by_station_cluster.png',
                       hline=1/24.)
plot_att_hist_by_label(B_rok[:, coldict['LocalHour']],
                       B_rok[:, coldict['Type']], time_range, nbins,
                       'Local hour', 'notecto_rok_hour_pdf_by_type.png',
                       hline=1/24.)
plot_att_hist_by_label(B_rok[:, coldict['LocalHour']],
                       B_labels_rok, time_range, nbins,
                       'Local hour', 'notecto_rok_hour_pdf_by_station_cluster.png',
                       hline=1/24.)

plot_att_hist_by_label_deconv(B_ant[:, coldict['LocalHour']],
                              X[:, coldict['LocalHour']],
                              B_labels_ant, clf.labels_, time_range, nbins,
                              'Local hour',
                              'notecto_ant_hour_pdf_by_station_cluster_deconv.png',
                              hline=1/24.)


B_wd = B[:, coldict['LocalWeekday']]
nbins = 7
time_range = (1, 8)
plot_att_hist_by_label(B_wd, B[:, coldict['Type']], time_range, nbins, 
                       'Local weekday', 'notecto_weekday_pdf_by_type.png',
                       hline=1/7.)
plot_att_hist_by_label(B_ant[:, coldict['LocalWeekday']],
                       B_ant[:, coldict['Type']], time_range, nbins,
                       'Local hour', 'notecto_ant_weekday_pdf_by_type.png',
                       hline=1/7.)
plot_att_hist_by_label(B_uni[:, coldict['LocalWeekday']],
                       B_uni[:, coldict['Type']], time_range, nbins,
                       'Local hour', 'notecto_uni_weekday_pdf_by_type.png',
                       hline=1/7.)
plot_att_hist_by_label(B_rok[:, coldict['LocalWeekday']],
                       B_rok[:, coldict['Type']], time_range, nbins,
                       'Local hour', 'notecto_rok_weekday_pdf_by_type.png',
                       hline=1/7.)


