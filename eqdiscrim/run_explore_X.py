import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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
    som_df[att_name].apply(np.log).plot.hist(20)
    eff_df[att_name].apply(np.log).plot.hist(20)
    plt.title(att_name)
    plt.savefig("%s.png" % att_name)
    fig.close()
