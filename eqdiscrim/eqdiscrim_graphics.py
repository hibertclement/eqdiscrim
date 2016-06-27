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
            df[att_name].apply(np.log10).plot.hist(20, alpha=0.5, normed=True,
                                                 color=c, label=label)
        plt.legend(loc='best')
        plt.xlabel('Log_10 (attribute)')
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
                    ax = plt.scatter(df[att_i], df[att_j], color=c, alpha=0.2,
                                    label=label)
                plt.xscale('log')
                plt.yscale('log')
                plt.legend(loc='best')
                plt.xlabel(att_i)
                plt.ylabel(att_j)
                plt.title("%s : %s vs %s" % (sta, att_j, att_i))
                plt.savefig(os.path.join(figdir, "%s_%s_%s_scatter.png" % (sta,
                                         att_j, att_i)))

def plot_lda(X_r2, y, classes, color_list, figdir, sta):

    plt.figure()
    for c, i, target_name in zip(color_list, range(len(classes)), classes):
        plt.scatter(X_r2[y == classes[i], 0], X_r2[y == classes[i], 1], c=c,
                    label=target_name)
    plt.legend(loc='best')
    plt.title('LDA on %s' % sta)
    plt.savefig(os.path.join(figdir, "%s_LDA.png" % (sta)))
    
 
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

def plot_learning_curve(train_sizes, train_scores, valid_scores, title, fname):

    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = 2 * np.std(train_scores, axis=1)
    test_scores_mean = np.mean(valid_scores, axis=1)
    test_scores_std = 2 * np.std(valid_scores, axis=1)

    fig = plt.figure()

    plt.ylim(0, 1.3)
    plt.title(title)
    plt.xlabel("Training examples")
    plt.ylabel("Score")

    plt.grid()
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")

    plt.savefig(fname)

def plot_confusion_matrix(cm, labels, title, fname, cmap=plt.cm.Blues):
    fig = plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels, rotation=45)
    plt.yticks(tick_marks, labels)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(fname)
