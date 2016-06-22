import pickle
import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import AffinityPropagation
import eqdiscrim_io as io

figdir = 'Figures'
if not os.path.exists(figdir):
    os.mkdir(figdir)

pd.set_option('mode.use_inf_as_null', True)

station_names = ["RVL", "FLR", "BOR", "BON", "SNE", "FJS", "CSS", "GPS", "GPN", "FOR"]
att_dir = "Attributes"

#event_types = ["Effondrement", "Sommital", "Local", "Teleseisme", "Onde sonore"]
event_types = ["Effondrement"]

for sta in station_names:
    # read attributes
    print "Treating station %s" % sta
    fnames = glob.glob(os.path.join(att_dir, 'X_%s_*_dataframe.dat' % (sta)))
    if len(fnames) == 0:
        continue
    print "Reading and concatenating %d dataframes" % len(fnames)
    X_df_full = io.read_and_cat_dataframes(fnames)

    # drop all lines containing nan
    print "Dropping all rows containing NaN"
    X_df_full.dropna(inplace=True)
    print X_df_full['EVENT_TYPE'].value_counts()

    # get the list of attributes we are interested in
    atts = X_df_full.columns[5:]
    att_list = list([att for att in atts])

    # extract the subsets according to type
    print "Extracting events according to type"
    df_list = []
    df_indexes_list = []
    for evtype in event_types:
        df = X_df_full[X_df_full['EVENT_TYPE'] == evtype]
        n = len(df)
        df_list.append(df)

    # put them all together
    df_to_cluster = pd.concat(df_list)

    # extract the training set
    print df_to_cluster['EVENT_TYPE'].value_counts()
    X_cluster = df_to_cluster[att_list].values
    Y_cluster = df_to_cluster['EVENT_TYPE'].values

    # Use affinity propagation clustering
    print "Clustering events"
    af = AffinityPropagation().fit(X_cluster)
    cluster_centers_indices = af.cluster_centers_indices_
    labels = af.labels_
    n_clusters_ = len(cluster_centers_indices)

    print n_clusters_

