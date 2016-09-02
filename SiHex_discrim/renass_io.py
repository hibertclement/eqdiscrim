import pandas as pd
import numpy as np
import pytz
from datetime import datetime

utc = pytz.utc


# files to be read (static for now - ugly but efficient)
renass_txt = '../static_catalogs/RENASS_stations_2014.txt'
stations_csv = '../static_catalogs/stations_fr.csv'


def construct_otime(row):
    date = row['OTIME']
    return utc.localize(datetime.strptime(date, "%Y-%m-%dT%H:%M:%S.%f"))


def read_BCSF_RENASS_cat(fname):
    names = list(["ID", "OTIME", "LAT", "LON", "PROF", "AUTEUR", "TYPE", "Mw"])
    df = pd.read_table(fname, delim_whitespace=True, header=None, names=names)
    df.set_index('ID', inplace=True)
    df['OTIME'] = df.apply(construct_otime, axis=1)

    return df

"""
def read_renass():

    # read the txt file using pandas
    s_pd = pd.read_table(renass_txt)

    names = list(["Code", "Lat", "Lon", "Elev(m)"])
    stations = s_pd[names].values

    return stations, names
"""


def construct_stime(row):

    s_date_parts = row['DATEDEB'].split('/')
    s_time_parts = row['HDEB'].split(':')

    # construct start time
    year = np.int(s_date_parts[0])
    month = np.int(s_date_parts[1])
    day = np.int(s_date_parts[2])
    hour = np.int(s_time_parts[0])
    minute = np.int(s_time_parts[1])
    seconds = np.int(np.floor(np.float(s_time_parts[2])))
    microseconds = np.int((np.float(s_time_parts[2]) - seconds) * 1e6)
    # make a datetime object by merging date and time
    try:
        stime = datetime(year, month, day, hour, minute, seconds,
                         microseconds, utc)
    except ValueError:
        if day == 0:
            # LDG sometimes uses day 01/00,00:00:00.0
            day = 1  # fix day to first of january
            stime = datetime(year, month, day, hour, minute, seconds,
                             microseconds, utc)
        else:
            raise

    return stime


def construct_etime(row):
    e_date_parts = row['DATEFIN'].split('/')
    e_time_parts = row['HFIN'].split(':')

    # construct end time
    year = np.int(e_date_parts[0])
    month = np.int(e_date_parts[1])
    day = np.int(e_date_parts[2])
    hour = np.int(e_time_parts[0])
    minute = np.int(e_time_parts[1])
    seconds = np.int(np.floor(np.float(e_time_parts[2])))
    microseconds = np.int((np.float(e_time_parts[2]) - seconds) * 1e6)
    # make a datetime object by merging date and time
    etime = datetime(year, month, day, hour, minute, seconds,
                     microseconds, utc)

    return etime


def read_stations_fr_dataframe():

    s_pd = pd.read_csv(stations_csv)
    s_pd['STIME'] = s_pd.apply(construct_stime, axis=1)
    s_pd['ETIME'] = s_pd.apply(construct_etime, axis=1)
    s_pd.drop(['DATEDEB', 'DATEFIN', 'HDEB', 'HFIN', 'NET'], axis=1,
              inplace=True)

    return s_pd

"""
def read_stations_fr():

    # read the csv file using pandas
    s_pd = pd.read_csv(stations_csv)
    names = list(["NOM", "LATITUDE", "LONGITUDE", "ELEV"])
    names_DT = list(["DATEDEB", "HDEB", "DATEFIN", "HFIN"])

    # read time information (coded as strings)
    DT = s_pd[names_DT].values
    nst, nd = DT.shape
    stime = np.empty(nst, dtype=object)
    etime = np.empty(nst, dtype=object)
    for i in xrange(nst):
        # decode date string
        s_date_parts = DT[i, 0].split('/')
        e_date_parts = DT[i, 2].split('/')
        s_time_parts = DT[i, 1].split(':')
        e_time_parts = DT[i, 3].split(':')

        # construct start time
        year = np.int(s_date_parts[0])
        month = np.int(s_date_parts[1])
        day = np.int(s_date_parts[2])
        hour = np.int(s_time_parts[0])
        minute = np.int(s_time_parts[1])
        seconds = np.int(np.floor(np.float(s_time_parts[2])))
        microseconds = np.int((np.float(s_time_parts[2]) - seconds) * 1e6)
        # make a datetime object by merging date and time
        try:
            stime[i] = datetime(year, month, day, hour, minute, seconds,
                                microseconds, utc)
        except ValueError:
            if day == 0:
                # LDG sometimes uses day 01/00,00:00:00.0
                day = 1  # fix day to first of january
                stime[i] = datetime(year, month, day, hour, minute, seconds,
                                    microseconds, utc)
            else:
                raise

        # construct end time
        year = np.int(e_date_parts[0])
        month = np.int(e_date_parts[1])
        day = np.int(e_date_parts[2])
        hour = np.int(e_time_parts[0])
        minute = np.int(e_time_parts[1])
        seconds = np.int(np.floor(np.float(e_time_parts[2])))
        microseconds = np.int((np.float(e_time_parts[2]) - seconds) * 1e6)
        # make a datetime object by merging date and time
        etime[i] = datetime(year, month, day, hour, minute, seconds,
                            microseconds, utc)

    # get the basic parameters
    s_tmp = s_pd[names].values
    nst, nd = s_tmp.shape

    stations = np.hstack((s_tmp, stime.reshape(nst, 1), etime.reshape(nst, 1)))

    # add the time values behind the basic ones and fix the names
    names.append("STIME")
    names.append("ETIME")

    return stations, names
"""
