import pandas as pd
import numpy as np
from obspy.core import read, UTCDateTime


def read_catalog(fname):

        names = list(['DATE', 'OTIME', 'TYPE' 'DUR', 'ONSET', 'SNR'])

        pd_cat = pd.read_excel(fname, skiprows=0, names=names)

        return pd_cat[names].values


def read_and_cut_events(cat, data_regex):

    nr, nc = cat.shape

    st_list = np.empty(nr, dtype=object)

    for i in xrange(nr):
        line = cat[i]
        time = UTCDateTime(line[0].replace(' ', 'T'))
        duration = line[2]
        onset = line[3]
        starttime = time + onset
        endtime = time + duration
        st = read(data_regex, starttime=starttime, endtime=endtime)
        st_list[i] = st

    return st_list
