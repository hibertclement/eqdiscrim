import pandas as pd
import numpy as np
from datetime import datetime
from dateutil import tz

utc = tz.gettz('UTC')


# files to be read (static for now - ugly but efficient - fix later)
sihex_xls = '../static_catalogs/SIHEXV2-inout-final.xlsx'
sihex_txt = '../static_catalogs/SIHEXV2-catalogue-final.txt'


def read_sihex_xls(inout=True):

    # read the excel file into pandas data frames
    eq_in = pd.read_excel(sihex_xls, sheetname=0)
    if inout:
        eq_out = pd.read_excel(sihex_xls, sheetname=1)

    # date-time reading
    DT_names = list(["DATE", "HEURE"])
    DT_in = eq_in[DT_names].values
    if inout:
        DT_out = eq_out[DT_names].values
        DT = np.vstack((DT_in, DT_out))
    else:
        DT = DT_in

    # date-time parsing into a single otime object
    nev, nd = DT.shape
    otime = np.empty(nev, dtype=object)
    for i in xrange(nev):
        # Date is encoded as a pandas imestamp
        year = DT[i, 0].year
        month = DT[i, 0].month
        day = DT[i, 0].day
        # time is encoded as a string
        time_parts = DT[i, 1].split(':')
        hour = np.int(time_parts[0])
        minute = np.int(time_parts[1])
        seconds = np.int(np.floor(np.float(time_parts[2])))
        microseconds = np.int((np.float(time_parts[2]) - seconds) * 1e6)
        # make a datetime object by merging date and time
        otime[i] = datetime(year, month, day, hour, minute, seconds,
                            microseconds,utc)

    # transform the pandas format to numpy format for later use in sklearn
    names = list(["ID", "LAT", "LON", "PROF", "Mw", "AUTEUR"])
    X_in = eq_in[names].values
    if inout:
        X_out = eq_out[names].values
        X_tmp = np.vstack((X_in, X_out)) 
    else:
        X_tmp = X_in

    # add the otime between "ID" and "LAT"
    nnames = len(names)
    X = np.hstack((X_tmp[:, 0].reshape(nev, 1), otime.reshape(nev, 1),
                   X_tmp[:, 1:].reshape(nev, nnames-1)))
    names.insert(1, "OTIME")

    # the current data file does not contain "ke" attributes
    # all the events are "ke"
    y = np.empty(nev, dtype='S2')
    y[...] = 'ke'

    return X, y, names
