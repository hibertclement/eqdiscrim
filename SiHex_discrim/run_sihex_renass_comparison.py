import os
import cat_io.sihex_io as io
import cat_io.renass_io as ior
import matplotlib.pyplot as plt

from pickle import load, dump
from sklearn import cluster

import preproc

catalog_dir = '../static_catalogs'
bcsf_cat = '../static_catalogs/catalogue_BCSF_RENASS_2012_2016.txt'
tidy_file_sihex = 'sihex_tidy_all_dataframe.dat'
tidy_file_bcsf = 'bcsf_tidy_all_dataframe.dat'
clust_file = 'clusteriser.dat'

read_tidy_sihex = True
read_tidy_renass = True
read_clust = True

df_bcsf = ior.read_BCSF_RENASS_cat(bcsf_cat)

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

    with open(os.path.join(catalog_dir, tidy_file_bcsf), 'wb') as f_:
        dump(df_bcsf, f_)
    

# print df['TYPE'].value_counts()
# print df.head()


# do clustering
if read_clust :
    print "Reading cluster"
    with open(clust_file, 'rb') as f_:
        clf = load(f_)
else :
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

# plotting
plt.figure()
df_0 = df_sihex[df_sihex['CLUST']==0]
df_1 = df_sihex[df_sihex['CLUST']==1]
df_2 = df_sihex[df_sihex['CLUST']==2]
plt.scatter(df_2['LON'], df_2['LAT'], color='blue', label='2')
plt.scatter(df_1['LON'], df_1['LAT'], color='green', label='1')
plt.scatter(df_0['LON'], df_0['LAT'], color='red', label='0')
plt.savefig('sihex_scatter.png')

plt.figure()
df_0 = df_bcsf[df_bcsf['CLUST']==0]
df_1 = df_bcsf[df_bcsf['CLUST']==1]
df_2 = df_bcsf[df_bcsf['CLUST']==2]
plt.scatter(df_2['LON'], df_2['LAT'], color='blue', label='2')
plt.scatter(df_1['LON'], df_1['LAT'], color='green', label='1')
plt.scatter(df_0['LON'], df_0['LAT'], color='red', label='0')
plt.savefig('bcsf_scatter.png')
