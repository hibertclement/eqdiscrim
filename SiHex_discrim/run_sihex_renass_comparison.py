import os
import numpy as np
import preproc
import sihex_io as io
import renass_io as ior
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
from pickle import load, dump
from sklearn import cluster


# ----------------------
# constants
# ----------------------

catalog_dir = '../static_catalogs'
bcsf_cat = '../static_catalogs/catalogue_BCSF_RENASS_2012_2016.txt'
tidy_file_sihex = 'sihex_tidy_all_dataframe.dat'
tidy_file_bcsf = 'bcsf_tidy_all_dataframe.dat'
clust_file = 'clusteriser.dat'

read_tidy_sihex = True
read_tidy_renass = True
read_clust = True

m = Basemap(llcrnrlon=-10., llcrnrlat=38., urcrnrlon=15., urcrnrlat=54.,
            resolution='i', projection='tmerc', lat_0=47., lon_0=1.)

colors = {'ke': 'blue',
          'km': 'red',
          'sm': 'red',
          'uk': 'cyan',
          'me': 'magenta',
          'ki': 'orange',
          'si': 'orange',
          'kr': 'brown',
          'sr': 'brown',
          'ls': 'lime',
          'fe': 'lime',
          '0': 'white',
          }

# ----------------------
# functions
# ----------------------


def split_by_clusters(df):
    df_0 = df[df['CLUST'] == 0]
    df_1 = df[df['CLUST'] == 1]
    df_2 = df[df['CLUST'] == 2]

    return df_0, df_1, df_2


def plot_clusters_geo(df, title, fname):

    plt.figure()
    df_0, df_1, df_2 = split_by_clusters(df)
    x2, y2 = m(df_2['LON'].values, df_2['LAT'].values)
    x1, y1 = m(df_1['LON'].values, df_1['LAT'].values)
    x0, y0 = m(df_0['LON'].values, df_0['LAT'].values)
    m.scatter(x2, y2, color='blue')
    m.scatter(x1, y1, color='green')
    m.scatter(x0, y0, color='red')
    m.drawcoastlines()
    plt.title(title)
    plt.savefig(fname)


def plot_types_geo(df, title, fname):

    labels = df['TYPE'].unique()
    plt.figure()

    for lab in labels:
        df_lab = df[df['TYPE'] == lab]
        x, y = m(df_lab['LON'].values, df_lab['LAT'].values)
        m.scatter(x, y, label=lab, color=colors[lab])

    m.drawcoastlines()
    plt.legend(loc='upper left')
    plt.title(title)
    plt.savefig(fname)


def plot_Mw_histograms(df, title, fname):

    plt.figure()
    fig, axes = plt.subplots(2, 1)
    fig.set_size_inches(6, 10)

    plt.sca(axes[0])
    n, bins, patches = plt.hist(df['Mw'].values, 10, normed=1, histtype='step',
                                color='black')
    plt.title(title)
    plt.ylabel('Probabillity density')

    plt.sca(axes[1])
    df_0, df_1, df_2 = split_by_clusters(df)
    n, bins, patches = plt.hist(df_0['Mw'].values, 10, normed=1,
                                histtype='step', color='red')
    n, bins, patches = plt.hist(df_1['Mw'].values, 10, normed=1,
                                histtype='step', color='green')
    n, bins, patches = plt.hist(df_2['Mw'].values, 10, normed=1,
                                histtype='step', color='blue')
    plt.xlabel('Magnitude Mw')
    plt.ylabel('Probabillity density')
    plt.savefig(fname)


def plot_GR(df, title, fname):

    plt.figure()

    fig, axes = plt.subplots(2, 1)
    fig.set_size_inches(6, 10)

    plt.sca(axes[0])

    log10N, mags = preproc.GutenbergRichter(df['Mw'].values, 0.0, 6.0, 0.2)
    plt.scatter(mags[0:-1], log10N, color='black')
    plt.ylabel('log10N')
    plt.xlabel('Magnitude Mw')

    plt.title(title)

    plt.sca(axes[1])
    df_0, df_1, df_2 = split_by_clusters(df)
    log10N, mags = preproc.GutenbergRichter(df_0['Mw'].values, 0.0, 6.0, 0.2)
    plt.scatter(mags[0:-1], log10N, color='red')
    log10N, mags = preproc.GutenbergRichter(df_1['Mw'].values, 0.0, 6.0, 0.2)
    plt.scatter(mags[0:-1], log10N, color='green')
    log10N, mags = preproc.GutenbergRichter(df_2['Mw'].values, 0.0, 6.0, 0.2)
    plt.scatter(mags[0:-1], log10N, color='blue')

    plt.ylabel('log10N')
    plt.xlabel('Magnitude Mw')

    plt.savefig(fname)


def plot_localtime_histograms(df, title, fname):

    plt.figure()
    bins = np.arange(25) - 0.5

    fig, axes = plt.subplots(2, 1)
    fig.set_size_inches(6, 10)

    plt.sca(axes[0])
    plt.hist(df['LOCAL_HOUR'].values, bins=bins, normed=1, histtype='step',
             color='black')
    plt.plot([-1, 25], [1/24., 1/24.], lw=2, color='black')
    plt.xlabel('Local hour')
    plt.ylabel('Probability density')
    plt.xlim(-1, 25)
    plt.ylim(0, 0.08)
    plt.title(title)

    plt.sca(axes[1])
    df_0, df_1, df_2 = split_by_clusters(df)
    plt.hist(df_0['LOCAL_HOUR'].values, bins=bins, normed=1, histtype='step',
             color='red')
    plt.hist(df_1['LOCAL_HOUR'].values, bins=bins, normed=1, histtype='step',
             color='green')
    plt.hist(df_2['LOCAL_HOUR'].values, bins=bins, normed=1, histtype='step',
             color='blue')
    plt.plot([-1, 25], [1/24., 1/24.], lw=2, color='black')
    plt.xlabel('Local hour')
    plt.ylabel('Probability density')
    plt.xlim(-1, 25)
    plt.ylim(0, 0.08)

    plt.savefig(fname)


def plot_weekday_histograms(df, title, fname):

    plt.figure()
    bins = np.arange(8) + 0.5

    fig, axes = plt.subplots(2, 1)
    fig.set_size_inches(6, 10)

    plt.sca(axes[0])

    plt.hist(df['WEEKDAY'].values, bins=bins, normed=1, histtype='step',
             color='black')
    plt.plot([0, 8], [1/7., 1/7.], lw=2, color='black')
    plt.xlabel('Local hour')
    plt.ylabel('Probability density')
    plt.xlim(0, 8)
    plt.ylim(0, 0.2)
    plt.title(title)

    plt.sca(axes[1])

    df_0, df_1, df_2 = split_by_clusters(df)
    plt.hist(df_0['WEEKDAY'].values, bins=bins, normed=1, histtype='step',
             color='red')
    plt.hist(df_1['WEEKDAY'].values, bins=bins, normed=1, histtype='step',
             color='green')
    plt.hist(df_2['WEEKDAY'].values, bins=bins, normed=1, histtype='step',
             color='blue')
    plt.plot([-1, 8], [1/7., 1/7.], lw=2, color='black')
    plt.xlabel('Local hour')
    plt.ylabel('Probability density')
    plt.xlim(0, 8)
    plt.ylim(0, 0.2)

    plt.savefig(fname)
    plt.close()


def plot_bar_stacked(df_workday, df_night_weekend, title, fname):

    labels = df_workday['TYPE'].unique()
    n = len(labels)
    ind = np.arange(n)
    width = 0.55

    counts = df_workday['TYPE'].value_counts()
    values1 = np.array([counts[lab] for lab in labels])
    counts = df_night_weekend['TYPE'].value_counts()
    values2 = np.array([counts[lab] for lab in labels])

    fig, axes = plt.subplots(1, 2)
    fig.set_size_inches(12, 5)
    plt.sca(axes[0])

    # plot values
    p1 = plt.bar(ind, values1, width, color='blue')
    p2 = plt.bar(ind, values2, width, color='red', bottom=values1)
    plt.ylabel('Number of events')
    plt.title(title)
    plt.xticks(ind+width/2., labels)
    plt.legend((p1[0], p2[0]), ('Workday', 'Night & weekend'))

    # plot fraction
    plt.sca(axes[1])
    f1 = 1.0*values1/(values1+values2)
    f2 = 1.0*values2/(values1+values2)
    p1 = plt.bar(ind, f1, width, color='blue')
    p2 = plt.bar(ind, f2, width, color='red', bottom=f1)
    plt.ylabel('Fraction')
    plt.title(title)
    plt.xticks(ind+width/2., labels)
    plt.axhline((5*12.0)/(7*24.), color='k', lw=2)

    plt.savefig(fname)
    plt.close()

# -------------------------------
# code starts here
# -------------------------------

if read_tidy_sihex:

    print "Loading from tidy file"
    # load from file
    with open(os.path.join(catalog_dir, tidy_file_sihex), 'rb') as f_:
        df_sihex = load(f_)
else:

    # Re-read the sihex files from the start
    df_sihex = io.read_all_sihex_files(catalog_dir)
    df_sihex = io.clean_sihex_data(df_sihex)
    df_sihex = io.mask_to_sihex_boundaries(catalog_dir, df_sihex)
    # keep only points within sihex boundaries
    df = df_sihex[df_sihex['IN_SIHEX']].copy()
    df.drop('IN_SIHEX', axis=1, inplace=True)
    # add distance to closest stations
    df_sihex = preproc.add_distance_to_closest_stations(df, 3)
    df_sihex = preproc.extract_local_hour_weekday(df_sihex)

    # dump to file
    with open(os.path.join(catalog_dir, tidy_file_sihex), 'wb') as f_:
        dump(df_sihex, f_)

if read_tidy_renass:
    print "Loading from tidy file"
    with open(os.path.join(catalog_dir, tidy_file_bcsf), 'rb') as f_:
        df_bcsf = load(f_)
else:

    df_bcsf = ior.read_BCSF_RENASS_cat(bcsf_cat)
    df_bcsf = io.mask_to_sihex_boundaries(catalog_dir, df_bcsf)
    # keep only points within sihex boundaries
    df = df_bcsf[df_bcsf['IN_SIHEX']].copy()
    df.drop('IN_SIHEX', axis=1, inplace=True)
    # add distance to closest stations
    df_bcsf = preproc.add_distance_to_closest_stations(df, 3)
    df_bcsf = preproc.extract_local_hour_weekday(df_bcsf)

    with open(os.path.join(catalog_dir, tidy_file_bcsf), 'wb') as f_:
        dump(df_bcsf, f_)

# do clustering
if read_clust:
    print "Reading cluster"
    with open(clust_file, 'rb') as f_:
        clf = load(f_)
else:
    print "Doing clustering"
    Xdist = df_sihex['DIST'].values
    Xdist = Xdist.reshape(-1, 1)
    clf = cluster.KMeans(init='k-means++', n_clusters=3, random_state=42)
    clf.fit(Xdist)

    with open(clust_file, 'wb') as f_:
        dump(clf, f_)

# predict cluster label
Xdist = df_sihex['DIST'].values
Xdist = Xdist.reshape(-1, 1)
df_sihex['CLUST'] = clf.predict(Xdist)

Xdist = df_bcsf['DIST'].values
Xdist = Xdist.reshape(-1, 1)
df_bcsf['CLUST'] = clf.predict(Xdist)

# get definite earthquakes only
df_sihex_ke = df_sihex[df_sihex['TYPE'] == 'ke']
df_bcsf_ke = df_bcsf[df_bcsf['TYPE'] == 'ke']

# get day / night split
df_sihex_workday = df_sihex[(df_sihex['LOCAL_HOUR'] >= 7.0) &
                            (df_sihex['LOCAL_HOUR'] <= 19.0) &
                            (df_sihex['WEEKDAY'] <= 6)]
df_sihex_night_weekend = df_sihex[(df_sihex['LOCAL_HOUR'] < 7.0) |
                                  (df_sihex['LOCAL_HOUR'] > 19.0) |
                                  (df_sihex['WEEKDAY'] > 6)]

df_bcsf_workday = df_bcsf[(df_bcsf['LOCAL_HOUR'] >= 7.0) &
                          (df_bcsf['LOCAL_HOUR'] <= 19.0) &
                          (df_bcsf['WEEKDAY'] <= 6)]
df_bcsf_night_weekend = df_bcsf[(df_bcsf['LOCAL_HOUR'] < 7.0) |
                                (df_bcsf['LOCAL_HOUR'] > 19.0) |
                                (df_bcsf['WEEKDAY'] > 6)]


# do plotting
print "Plotting"
plot_clusters_geo(df_sihex, 'SIHEX - distance to station clusters',
                  'figures/sihex_cluster.png')
plot_clusters_geo(df_bcsf, 'BCSF - distance to station clusters',
                  'figures/bcsf_cluster.png')

plot_types_geo(df_sihex, 'SiHex', 'figures/sihex_types.png')
plot_types_geo(df_bcsf, 'BCSF', 'figures/bcsf_types.png')

plot_Mw_histograms(df_sihex_ke, 'SIHEX - Mw', 'figures/sihex_Mw.png')
plot_Mw_histograms(df_bcsf_ke, 'BCSF - Mw', 'figures/bcsf_Mw.png')

plot_GR(df_sihex_ke, 'GR - SiHex', 'figures/sihex_GR.png')
plot_GR(df_bcsf_ke, 'GR - BCSF', 'figures/bcsf_GR.png')

plot_localtime_histograms(df_sihex_ke, 'SiHex - ke',
                          'figures/sihex_ke_localtime.png')
plot_localtime_histograms(df_bcsf_ke, 'BCSF - ke',
                          'figures/bcsf_ke_localtime.png')

plot_weekday_histograms(df_sihex_ke, 'SiHex - ke',
                        'figures/sihex_ke_weekday.png')
plot_weekday_histograms(df_bcsf_ke, 'BCSF - ke',
                        'figures/bcsf_ke_weekday.png')

plot_bar_stacked(df_sihex_workday, df_sihex_night_weekend,
                 'SIHEX day/night/weekend',
                 'figures/sihex_day_night_weekend.png')
plot_bar_stacked(df_bcsf_workday, df_bcsf_night_weekend,
                 'BCSF day/night/weekend',
                 'figures/bcsf_day_night_weekend.png')
