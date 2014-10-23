import pandas as pd
import numpy as np
from datetime import datetime
from dateutil import tz

utc = tz.gettz('UTC')


# files to be read (static for now - ugly but efficient - fix later)
sihex_xls = '../static_catalogs/SIHEXV2-inout-final.xlsx'
sihex_txt = '../static_catalogs/SIHEXV2-catalogue-final.txt'
notecto_lst = '../static_catalogs/no_tecto.lst'


def read_notecto_lst():

    # use the same names as for the tectonic events
    names = list(["ID", "DATE", "HEURE", "LAT", "LON", "PROF", "AUTEUR", "TYPE", "Mw"])

    # read the .lst file into pandas data frame
    pd_ev = pd.read_table(notecto_lst, sep='\s+', header=None, names=names)

    # date-time reading
    DT_names = list(["DATE", "HEURE"])
    DT = pd_ev[DT_names].values

    # date-time parsing into a single otime object
    nev, nd = DT.shape
    otime = np.empty(nev, dtype=object)
    for i in xrange(nev):
        # Both date and time are encoded as strings
        date_parts = DT[i, 0].split('/')
        year = np.int(date_parts[0])
        month = np.int(date_parts[1])
        day = np.int(date_parts[2])
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
    X_tmp = pd_ev[names].values

    # add the otime between "ID" and "LAT"
    nnames = len(names)
    X = np.hstack((X_tmp[:, 0].reshape(nev, 1), otime.reshape(nev, 1),
                   X_tmp[:, 1:].reshape(nev, nnames-1)))
    names.insert(1, "OTIME")

    # the current data file does not contain "ke" attributes
    # all the events are "ke"
    y = pd_ev["TYPE"].values

    # PROBLEM WITH EVENTS in the out list : some are not near France
    ipb=[]
    # these events are no-where near the SiHex boundaries
    ipb.append(np.where(X[:, 0]==625133))
    ipb.append(np.where(X[:, 0]==621845))
    ipb.append(np.where(X[:, 0]==675405))
    ipb.append(np.where(X[:, 0]==217446))
    ipb.append(np.where(X[:, 0]==203905))
    ipb.append(np.where(X[:, 0]==659188))
    ipb.append(np.where(X[:, 0]==659107))
    ipb.append(np.where(X[:, 0]==659119))
    ipb.append(np.where(X[:, 0]==659512))
    ipb.append(np.where(X[:, 0]==659334))
    ipb.append(np.where(X[:, 0]==659122))
    ipb.append(np.where(X[:, 0]==658973))
    ipb.append(np.where(X[:, 0]==658990))
    ipb.append(np.where(X[:, 0]==620725))
    ipb.append(np.where(X[:, 0]==622160))
    ipb.append(np.where(X[:, 0]==285626))
    ipb.append(np.where(X[:, 0]==659575))

    # clean up
    X = np.delete(X, (ipb), axis=0)
    y = np.delete(y, (ipb), axis=0)


    return X, y, names


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

    # PROBLEM WITH EVENTS in the out list : some are not near France
    ipb=[]
    if inout:
        # These events are in the OUT part and are too far from France
        ipb.append(np.where(X[:, 0]==255001))
        ipb.append(np.where(X[:, 0]==180307))
        ipb.append(np.where(X[:, 0]==253079))
        ipb.append(np.where(X[:, 0]==256577))
        ipb.append(np.where(X[:, 0]==180664))
        ipb.append(np.where(X[:, 0]==180050))
        ipb.append(np.where(X[:, 0]==177133))
        ipb.append(np.where(X[:, 0]==179219))
        ipb.append(np.where(X[:, 0]==177792))
        ipb.append(np.where(X[:, 0]==313098))
        ipb.append(np.where(X[:, 0]==670271))
        ipb.append(np.where(X[:, 0]==640834))
        ipb.append(np.where(X[:, 0]==658151))
        ipb.append(np.where(X[:, 0]==657909))
        ipb.append(np.where(X[:, 0]==255946))
    # this next event has no depth
    ipb.append(np.where(X[:, 0]==640818))
    # this next event is rather too deep
    ipb.append(np.where(X[:, 0]==159405))
    X = np.delete(X, (ipb), axis=0)
    y = np.delete(y, (ipb), axis=0)

    return X, y, names
