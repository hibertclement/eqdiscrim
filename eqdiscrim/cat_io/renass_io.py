import pandas as pd
import numpy as np


# files to be read (static for now - ugly but efficient)
renass_txt = '../static_catalogs/RENASS_stations_2014.txt'
stations_csv = '../static_catalogs/stations_fr.csv'

def read_renass():

    # read the txt file using pandas
    s_pd = pd.read_table(renass_txt)

    names = list(["Code", "Lat", "Lon", "Elev(m)"])
    stations = s_pd[names].values

    return stations, names

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
    for i in xrange(nev):
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
        stime[i] = datetime(year, month, day, hour, minute, seconds,

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


    # get the basic parameters
    s_tmp = s_pd[names].values

    # add the time values behind the basic ones and fix the names

    return stations, names

