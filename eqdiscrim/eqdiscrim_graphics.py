import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def plot_histograms(df_list, att_list, color_list, figdir, sta):
    for att_name in att_list:
        print sta, att_name
        fig = plt.figure()
        for i in xrange(len(df_list)):
            df = df_list[i]
            c = color_list[i]
            label = df['EVENT_TYPE'].unique()[0]
            df[att_name].apply(np.log).plot.hist(20, alpha=0.5, normed=True,
                                                 color=c, label=label)
        plt.legend()
        plt.title("%s - %s" % (sta, att_name))
        plt.savefig(os.path.join(figdir, "%s_%s_hist.png" % (sta, att_name)))

def plot_scatterplots(df_list, att_list, color_list, figdir, sta):
    n_att = len(att_list)
    for i in xrange(n_att):
        for j in xrange(n_att):
            if j > i:
                att_i = att_list[i]
                att_j = att_list[j]
                print sta, att_i, att_j
                fig = plt.figure()
                for i_df in xrange(len(df_list)):
                    df = df_list[i_df]
                    c = color_list[i_df]
                    label = df['EVENT_TYPE'].unique()[0]
                    if i_df == 0:
                        ax = df.plot.scatter(x=att_i, y=att_j, color=c, alpha=0.2, label=label,
                                         logx=True, logy=True)
                    else:
                        df.plot.scatter(x=att_i, y=att_j, color=c, alpha=0.2, label=label,
                                         logx=True, logy=True, ax=ax)
                plt.xlabel(att_i)
                plt.ylabel(att_j)
                plt.title("%s : %s vs %s" % (sta, att_j, att_i))
                plt.savefig(os.path.join(figdir, "%s_%s_%s_scatter.png" % (sta,
                                         att_j, att_i)))

 
def plot_att_timeseries(df_list, att_list, color_list, figdir, sta):
   # time plots
    n_att = len(att_list)
    for i in xrange(n_att):
        att = att_list[i]
        print "Time plot", sta, att
        fig = plt.figure()
        for i_df in xrange(len(df_list)):
            df = df_list[i_df]
            c = color_list[i_df]
            label = df['EVENT_TYPE'].unique()[0]
            ts = df['WINDOW_START'].copy()
            df.loc[:, 'WINDOW_TS'] = pd.to_datetime(ts)
            if i_df == 0:
                ax = df.plot(x='WINDOW_TS', y=att, color=c, alpha=0.5, label=label,
                            logy=True)
            else:
                df.plot(x='WINDOW_TS', y=att, color=c, alpha=0.5, label=label,
                            logy=True, ax=ax)
        plt.xlabel('Date')
        plt.ylabel(att)
        plt.title("%s : %s vs date" % (sta, att))
        plt.savefig(os.path.join(figdir, "%s_%s_date.png" % (sta, att)))

 
