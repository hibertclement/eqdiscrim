import pandas as pd
import numpy as np
from os.path import join
from scipy.stats import kurtosis
from obspy.core import read
from obspy.signal.filter import envelope
from obspy.signal.util import smooth
from obspy.signal.trigger import triggerOnset

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
    names = ['Kurtosis', 'MaxMean', 'DomT', 'Dur']

    X = np.empty((nev, len(names)), dtype=float)
    # for each event
    for i in xrange(nev):
        otime = X_orig[i]
        fname = "%s_*MSEED"%(otime)
        st = read(join(syn_dir, fname))
        tr = st[0]
        starttime = tr.stats.starttime
        dt = tr.stats.delta
        # get start and end of signal
        i_start, i_end = start_end(tr)
        # trim data
        tr.trim(starttime=starttime+i_start*dt, endtime=starttime+i_end*dt)
        # now calculate the attributes
        kurt = kurtosis(tr.data)
        env = envelope(tr.data)
        max_mean = np.max(env) / np.mean(env)
        dur = (i_end - i_start) * dt
        X[i, 0] = kurt
        X[i, 1] = max_mean
        X[i, 2] = dominant_period(st)
        X[i, 3] = dur

    return X, y, names

def start_end(tr):
    """
    Returns start and end times of signal using signal 2 noise ratio
    """

    # set noise level as 5th percentile on envelope amplitudes
    env = envelope(tr.data)
    env = smooth(env, 100)
    noise_level = np.percentile(env, 5.0)

    # trigger
    t_list = triggerOnset(env, 1.5*noise_level, 1.5*noise_level)
    i_start = t_list[0][0]
    i_end = t_list[0][1]

    return i_start, i_end
    

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

    D[0] = tr_diff.data[0]**2
    X[0] = tr.data[0]**2
    T[0] = 2 * np.pi * np.sqrt(X[0]/D[0])

    for j in xrange(npts-1):
        i = j+1
        D[i] = alpha * D[i-1] + tr_diff[i]**2
        X[i] = alpha * X[i-1] + tr[i]**2
        T[i] = 2 * np.pi * np.sqrt(X[i]/D[i])

    return np.median(T)
    
