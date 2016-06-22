import pickle
import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import eqdiscrim_io as io
import eqdiscrim_graphics as gr

do_histograms = False
do_scatterplots = False
do_timeplots = False
do_radviz = False

figdir = 'Figures'
if not os.path.exists(figdir):
    os.mkdir(figdir)

pd.set_option('mode.use_inf_as_null', True)

# station_names = ["RVL", "FLR", "BOR", "BON", "SNE", "FJS", "CSS", "GPS", "GPN", "FOR"]
station_names = ["RVL", "BOR"]
att_dir = "Attributes"

best_atts = list(['RappMaxStd', 'Duration', 'RappMaxMean', 'KurtoEnv',
                  'RappMaxMedian', 'AsDec'])
 
event_types = ["Effondrement", "Sommital", "Local", "Teleseisme", "Onde sonore"]
color_list = ['b', 'g', 'r', 'y', 'cyan']

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

    all_atts = X_df_full.columns[5:]

    # extract the subsets according to type
    print "Extracting events according to type"
    df_list = []
    for evtype in event_types:
        df = X_df_full[X_df_full['EVENT_TYPE'] == evtype]
        df_list.append(df)

    print X_df_full['EVENT_TYPE'].value_counts()

    # histograms
    if do_histograms:
        print "Plotting histograms for %d classes and %d attributes" %\
              (len(df_list), len(best_atts))
        gr.plot_histograms(df_list, best_atts, color_list, figdir, sta)

    # scatter plots
    if do_scatterplots:
        print "Plotting scatter plots for %d classes and %d attributes" %\
            (len(df_list), len(best_atts))
        gr.plot_scatterplots(df_list, best_atts, color_list, figdir, sta)

    # time plots
    if do_timeplots:
        print "Plotting timeseries plots for %d classes and %d attributes" %\
              (len(df_list), len(best_atts))
        gr.plot_att_timeseries(df_list, best_atts, color_list, figdir, sta)

    # doing radviz plot
    if do_radviz:
        print "Doing radviz plot"
        best_atts.append('EVENT_TYPE')
        all_df = pd.concat(df_list)
        df_best = all_df[best_atts]   
        plt.figure()
        pd.tools.plotting.radviz(df_best, 'EVENT_TYPE')
        plt.title("%s - Radviz plot" % sta)
        plt.savefig(os.path.join(figdir, "%s_radviz.png" % sta))
