import pickle
import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import eqdiscrim_io as io
import eqdiscrim_graphics as gr

# ---------------------
# SWITCHES
do_histograms = True
do_scatterplots = False
do_lda_plots = False
do_timeplots = False
do_radviz = False
# ---------------------

# ---------------------
# Parameters
station_names = ["RVL", "BOR"]
att_dir = "Attributes"
figdir = 'Figures'
color_list = ['b', 'g', 'r', 'y', 'c', 'm', 'k', 'pink']
event_types = ['Teleseisme', 'Regional', 'Local', 'Sommital', 'Profond',
               'Effondrement', 'Phase T', 'Onde sonore']
best_atts_fname = 'best_attributes.dat'
# ---------------------

# ---------------------
# Code starts here

pd.set_option('mode.use_inf_as_null', True)

if not os.path.exists(figdir):
    os.mkdir(figdir)

# make a different plot for eah station
for sta in station_names:

    print "\nTreating station %s" % sta

    # read and cat all relevant dataframes
    fnames = glob.glob(os.path.join(att_dir, 'X_*_%s_*_dataframe.dat' % (sta)))
    if len(fnames) == 0:
        continue
    print "Reading and concatenating %d dataframes" % len(fnames)
    X_df_full = io.read_and_cat_dataframes(fnames)

    # drop all lines containing nan
    print "Dropping all rows containing NaN"
    X_df_full.dropna(inplace=True)

    # if have best_atts, then use them
    if os.path.exists(best_atts_fname):
        f_ = open(best_atts_fname, 'r')
        att_dict = pickle.load(f_)
        all_atts = att_dict[sta][0:3]
        f_.close()
    else:
        all_atts = X_df_full.columns[5:]

    # extract the subsets according to type
    print "Extracting events according to type"
    df_list = []
    for evtype in event_types:
        df = X_df_full[X_df_full['EVENT_TYPE'] == evtype]
        df_list.append(df)

    # histograms
    if do_histograms:
        print "Plotting histograms for %d classes and %d attributes" %\
              (len(df_list), len(all_atts))
        gr.plot_histograms(df_list, all_atts, color_list, figdir, sta)

    # scatter plots
    if do_scatterplots:
        print "Plotting scatter plots for %d classes and %d attributes" %\
            (len(df_list), len(all_atts))
        gr.plot_scatterplots(df_list, all_atts, color_list, figdir, sta)

    # do linear discriminant analysis plot for single station
    if do_lda_plots:
        print "Doing and plotting LDA on %s" % sta
        lda_df = pd.concat(df_list)
        # only keep the 
        for a in lda_df.columns[5:].values:
            if a not in all_atts:
                lda_df.drop(a, axis=1, inplace=True)
            else:
                lda_df[a].apply(np.log10)
        X = lda_df[lda_df.columns[5:]].values
        y = lda_df['EVENT_TYPE'].values
        lda = LinearDiscriminantAnalysis(n_components=2)
        X_r2 = lda.fit(X, y).transform(X)
        gr.plot_lda(X_r2, y, lda.classes_, color_list, figdir, sta)


        pass
    # time plots
    if do_timeplots:
        print "Plotting timeseries plots for %d classes and %d attributes" %\
              (len(df_list), len(all_atts))
        gr.plot_att_timeseries(df_list, all_atts, color_list, figdir, sta)

    # doing radviz plot
    if do_radviz:
        print "Doing radviz plot"
        all_atts.append('EVENT_TYPE')
        all_df = pd.concat(df_list)
        df_all = all_df[all_atts]   
        plt.figure()
        pd.tools.plotting.radviz(df_all, 'EVENT_TYPE')
        plt.title("%s - Radviz plot" % sta)
        plt.savefig(os.path.join(figdir, "%s_radviz.png" % sta))
