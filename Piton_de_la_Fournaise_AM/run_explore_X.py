import pickle
import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

figdir = 'Figures'
if not os.path.exists(figdir):
    os.mkdir(figdir)

pd.set_option('mode.use_inf_as_null', True)

station_names = ["RVL", "FLR", "BOR", "BON", "SNE", "FJS", "CSS", "GPS", "GPN", "FOR"]
att_dir = "Attributes"

best_atts = list(['FCentroid', 'gammas', 'DurOverAmp', 'Fquart3', 'Duration',
             'gamma1', 'Fquart1'])
for sta in station_names:
    fnames = glob.glob(os.path.join(att_dir, 'X_%s_*_dataframe.dat' % (sta)))
    if len(fnames) == 0:
        continue
    for fname in fnames:
        f_ = open(fname, 'r')
        X_df = pickle.load(f_)
        f_.close()
        if fname is fnames[0]:
            X_df_full = X_df
        else:
            X_df_full = X_df_full.append(X_df, ignore_index=False)

    X_df_full.dropna(inplace=True)
    print X_df_full.head()
    print sta
    print X_df_full['EVENT_TYPE'].value_counts()
    eff_df = X_df_full[X_df_full['EVENT_TYPE'] == 'Effondrement']
    som_df = X_df_full[X_df_full['EVENT_TYPE'] == 'Sommital']
    loc_df = X_df_full[X_df_full['EVENT_TYPE'] == 'Local']

    # histograms
    for att_name in best_atts:
        print sta, att_name
        fig = plt.figure()
        eff_df[att_name].apply(np.log).plot.hist(20, alpha=0.5, normed=True,
                                                 label='EFF')
        som_df[att_name].apply(np.log).plot.hist(20, alpha=0.5, normed=True,
                                                 label='SOM')
        loc_df[att_name].apply(np.log).plot.hist(20, alpha=0.5, normed=True,
                                                 label='LOC')
        plt.title("%s - %s" % (sta, att_name))
        plt.savefig(os.path.join(figdir, "%s_%s_hist.png" % (sta, att_name)))

    # scatter plots
    n_best = len(best_atts)
    for i in xrange(n_best):
        for j in xrange(n_best):
            if j > i:
                print sta, best_atts[i], best_atts[j]
                fig = plt.figure()
                ax = eff_df.plot.scatter(x=best_atts[i], y=best_atts[j],
                                         color='b', alpha=0.2, label='EFF',
                                         logx=True, logy=True)
                som_df.plot.scatter(x=best_atts[i], y=best_atts[j], color='g',
                                    alpha=0.2, ax=ax, label='SOM', logx=True,
                                    logy=True)
                loc_df.plot.scatter(x=best_atts[i], y=best_atts[j], color='r',
                                    alpha=0.2, ax=ax, label='LOC', logx=True,
                                    logy=True)
                plt.xlabel(best_atts[i])
                plt.ylabel(best_atts[j])
                plt.title("%s : %s vs %s" % (sta, best_atts[j], best_atts[i]))
                plt.savefig(os.path.join(figdir, "%s_%s_%s_scatter.png" % (sta,
                                         best_atts[j], best_atts[i])))

    # time plots
    for i in xrange(n_best):
        print "Time plot", sta, best_atts[i]
        fig = plt.figure()
        ts = eff_df['WINDOW_START'].copy()
        eff_df.loc[:, 'WINDOW_TS'] = pd.to_datetime(ts)
        ax = eff_df.plot(x='WINDOW_TS', y=best_atts[i], color='b', alpha=0.5, label='EFF',
                            logy=True)
        ts = som_df['WINDOW_START'].copy()
        som_df.loc[:, 'WINDOW_TS'] = pd.to_datetime(ts)
        som_df.plot(x='WINDOW_TS', y=best_atts[i], color='g', alpha=0.5, label='SOM',
                            logy=True, ax=ax)
        ts = loc_df['WINDOW_START'].copy()
        loc_df.loc[:, 'WINDOW_TS'] = pd.to_datetime(ts)
        loc_df.plot(x='WINDOW_TS', y=best_atts[i], color='r', alpha=0.5, label='LOC',
                            logy=True, ax=ax)
        plt.xlabel('Date')
        plt.ylabel(best_atts[i])
        plt.title("%s : %s vs date" % (sta, best_atts[i]))
        plt.savefig(os.path.join(figdir, "%s_%s_date.png" % (sta, best_atts[i])))

    print "Doing radviz plot"
    best_atts.append('EVENT_TYPE')
    all_df = pd.concat([eff_df, som_df, loc_df])
    df_best = all_df[best_atts]   
    plt.figure()
    pd.tools.plotting.radviz(df_best, 'EVENT_TYPE')
    plt.title("%s - Radviz plot" % sta)
    plt.savefig(os.path.join(figdir, "%s_radviz.png" % sta))
