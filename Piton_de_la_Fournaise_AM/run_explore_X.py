import pickle
import os
import glob
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import eqdiscrim_io as io
import eqdiscrim_graphics as gr


pd.set_option('mode.use_inf_as_null', True)


def make_att_matrix(all_atts_dict, cfg):

    ncols = len(cfg.station_names) * cfg.n_best_atts
    nrows = len(all_atts_dict.keys())
    matrix = np.zeros((nrows, ncols), dtype=int)

    sta_labels = cfg.station_names[:]
    for key in all_atts_dict.keys():
        if key not in cfg.station_names:
            sta_labels.append(key)

    att_names = []
    for sta in cfg.station_names:
        att_list = all_atts_dict[sta]
        new_att_list = ['%s_%s' % (sta, a) for a in att_list]
        all_atts_dict[sta] = new_att_list
        att_names += new_att_list
            
    for sta_lab in sta_labels:
        i = sta_labels.index(sta_lab)
        for a in all_atts_dict[sta_lab]:
            j = att_names.index(a)
            matrix[i, j] = 1

    return matrix, sta_labels, att_names


def run_explore_data(args):

    cfg = io.Config(args.config_file)

    if cfg.do_translation:
        n_names = len(cfg.event_types)
        tr_dict = {}
        for i in xrange(n_names):
            tr_dict[cfg.event_types[i]] = cfg.event_types_translated[i]

    if not os.path.exists(cfg.figdir):
        os.mkdir(cfg.figdir)

    all_atts = {}
    # make a different plot for eah station
    for sta in cfg.station_names:

        print "\nTreating station %s" % sta

        # read and cat all relevant dataframes
        fnames = glob.glob(os.path.join(cfg.att_dir, 'X_*_%s_*_dataframe.dat' %
                                       (sta)))
        if len(fnames) == 0:
            continue
        print "Reading and concatenating %d dataframes" % len(fnames)
        X_df_full = io.read_and_cat_dataframes(fnames)

        # drop all lines containing nan
        print "Dropping all rows containing NaN"
        X_df_full.dropna(inplace=True)

        # if have best_atts, then use them
        if os.path.exists(cfg.best_atts_fname):
            with open(cfg.best_atts_fname, 'r') as f_:
                att_dict = pickle.load(f_)
                all_atts[sta] = att_dict[sta]
        else:
            all_atts[sta] = X_df_full.columns[5:]

        # extract the subsets according to type
        print "Extracting events according to type"
        df_list = []
        for evtype in cfg.event_types:
            df = X_df_full[X_df_full['EVENT_TYPE'] == evtype]
            if cfg.do_translation:
                df['EVENT_TYPE'] = tr_dict[evtype]
            df_list.append(df)

        # histograms
        if cfg.do_histograms:
            print "Plotting histograms for %d classes and %d attributes" %\
                  (len(df_list), len(all_atts[sta]))
            gr.plot_histograms(df_list, all_atts[sta], cfg.color_list, cfg.figdir, sta)

        # scatter plots
        if cfg.do_scatterplots:
            print "Plotting scatter plots for %d classes and %d attributes" %\
                  (len(df_list), len(all_atts[sta]))
            gr.plot_scatterplots(df_list, all_atts[sta], cfg.color_list, cfg.figdir, sta)

        # do linear discriminant analysis plot for single station
        if cfg.do_lda_plots:
            print "Doing and plotting LDA on %s" % sta
            lda_df = pd.concat(df_list)
            # only keep the 
            for a in lda_df.columns[5:].values:
                if a not in all_atts[sta]:
                    lda_df.drop(a, axis=1, inplace=True)
                else:
                    lda_df[a].apply(np.log10)
            X = lda_df[lda_df.columns[5:]].values
            y = lda_df['EVENT_TYPE'].values
            lda = LinearDiscriminantAnalysis(n_components=2)
            X_r2 = lda.fit(X, y).transform(X)
            gr.plot_lda(X_r2, y, lda.classes_, cfg.color_list, cfg.figdir, sta)


        # time plots
        if cfg.do_timeplots:
            print "Plotting timeseries plots for %d classes and %d attributes" %\
                (len(df_list), len(all_atts[sta]))
            gr.plot_att_timeseries(df_list, all_atts[sta], cfg.color_list, cfg.figdir, sta)

        # doing radviz plot
        if cfg.do_radviz:
            print "Doing radviz plot"
            all_atts[sta].append('EVENT_TYPE')
            all_df = pd.concat(df_list)
            df_all = all_df[all_atts[sta]]   
            plt.figure()
            pd.tools.plotting.radviz(df_all, 'EVENT_TYPE')
            plt.title("%s - Radviz plot" % sta)
            plt.savefig(os.path.join(cfg.figdir, "%s_radviz.png" % sta))

    if cfg.do_att_matrix:
        matrix, row_names, col_names = make_att_matrix(att_dict, cfg)
        gr.plot_att_matrix(matrix, row_names, col_names, cfg.figdir)

if __name__ == '__main__':

    # set up parser
    cl_parser = argparse.ArgumentParser(description='Explore and plot figures')
    cl_parser.add_argument('config_file', help='eqdiscrim configuration file')

    args = cl_parser.parse_args()
    run_explore_data(args)

