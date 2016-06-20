import pickle
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

figdir = 'Figures'
if not os.path.exists(figdir):
    os.mkdir(figdir)

pd.set_option('mode.use_inf_as_null', True)

fname = 'X_dataframe.dat'
f_ = open(fname, 'r')
X_df = pickle.load(f_)
f_.close()

print X_df['EVENT_TYPE'].value_counts()
eff_df = X_df[X_df['EVENT_TYPE'] == 'Effondrement']
som_df = X_df[X_df['EVENT_TYPE'] == 'Sommital']

atts = X_df.columns[5:]
for att_name in atts:
    print att_name
    fig = plt.figure()
    eff_df[att_name].apply(np.log).plot.hist(20)
    som_df[att_name].apply(np.log).plot.hist(20)
    plt.title(att_name)
    plt.savefig(os.path.join(figdir, "%s.png" % att_name))
