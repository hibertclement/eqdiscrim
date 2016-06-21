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
    print sta
    print X_df_full['EVENT_TYPE'].value_counts()
    eff_df = X_df_full[X_df_full['EVENT_TYPE'] == 'Effondrement']
    som_df = X_df_full[X_df_full['EVENT_TYPE'] == 'Sommital']
    loc_df = X_df_full[X_df_full['EVENT_TYPE'] == 'Local']

    atts = X_df_full.columns[5:]
    for att_name in atts:
        print att_name
        fig = plt.figure()
        eff_df[att_name].apply(np.log).plot.hist(20, alpha=0.5, normed=True)
        som_df[att_name].apply(np.log).plot.hist(20, alpha=0.5, normed=True)
        loc_df[att_name].apply(np.log).plot.hist(20, alpha=0.5, normed=True)
        plt.title(att_name)
        plt.savefig(os.path.join(figdir, "%s_%s.png" % (sta, att_name)))
