import pandas as pd
import numpy as np
from os.path import join
from scipy.stats import kurtosis
from obspy.core import read
from obspy.signal.filter import envelope

syn_dir = '../synthetic_data'
syncat = join(syn_dir, 'syn_catalog.txt')

def read_syncat():

    names = list(["Otime", "Type"])
    s_pd = pd.read_table(syncat, sep='\s+', header=0, names=names)

    X = s_pd['Otime'].values
    y = s_pd['Type'].values

    return X, y, ['Otime']

def proc_syn_data():

    X_orig, y, names = read_syncat()

    nev = len(y)
    names = ['Kurtosis', 'MaxMean', 'DomT']

    X = np.empty((nev, len(names)), dtype=float)
    # for each event
    for i in xrange(nev):
        otime = X_orig[i]
        fname = "%s_*MSEED"%(otime)
        st = read(join(syn_dir, fname))
        tr = st[0]
        kurt = kurtosis(tr.data)
        env = envelope(tr.data)
        max_mean = np.max(env) / np.mean(env)
        X[i, 0] = kurt
        X[i, 1] = max_mean
        X[i, 2] = dominant_period(st)

    return X, y, names


def dominant_period(st):
    """
    Returns dominant period in the Nakamura sense
    """

    # differentiate the waveform
    st_diff = st.copy()
    st_diff.differentiate()

    # get the traces of the original and differentiated waveform
    tr = st[0]
    tr_diff = st_diff[0]
    npts = tr.stats.npts

    # set up the constants and empty space
    alpha = 0.999
    D = np.empty(npts, dtype=float)
    X = np.empty(npts, dtype=float)
    T = np.empty(npts, dtype=float)

    D[0] = tr_diff[0]**2
    X[0] = tr[0]**2
    T[0] = 2 * np.pi * np.sqrt(X[0]/D[0])

    for j in xrange(npts-1):
        i = j+1
        D[i] = alpha * D[i-1] + tr_diff[i]**2
        X[i] = alpha * X[i-1] + tr[i]**2
        T[i] = 2 * np.pi * np.sqrt(X[i]/D[i])

    return np.median(T)
    
