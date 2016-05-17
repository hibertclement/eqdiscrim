import locale
import pandas as pd
import numpy as np
from obspy.core.utcdatetime import UTCDateTime


def read_ovpf_cat(filename):
    """
    reading of the catalogue ovpf
    """
    
    # read catalog using pandas     
    cat_tmp = pd.read_csv(filename, sep=';')

    # be careful : the csv file contains durations as 5,6 with commas
    # regardless of locale, set LC_NUMERIC for France
    locale.setlocale(locale.LC_NUMERIC, 'fr_FR')
    
    # change commas to . in duration
    dur = cat_tmp["Duration"].apply(locale.atof)
    
    # turn the pandas dataframe to a numpy array
    names = list(["#YYYYmmdd HHMMSS.ss", "Duration", "Type", "Operator"])
    cat = cat_tmp[names].values
    nev, bo = cat.shape
    
    # extract the timing information and turn into UTCDateTime
    dt = cat[:, 0]
    time = np.empty(nev, dtype=object)
    for i in xrange(nev):
        t = UTCDateTime(dt[i])
        time[i] = t
      
    # pack the data back together  
    data = np.hstack((time.reshape(nev, 1), dur.reshape(nev, 1), cat[:, 2:]))
    
    return data
    