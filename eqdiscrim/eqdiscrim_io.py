# -*- coding: utf-8 -*-

import os
import locale
import pytz
import urllib
import tempfile 
import pandas as pd
import numpy as np
from obspy import UTCDateTime, read
from obspy.clients.fdsn import Client
from datetime import timedelta

# client = Client("http://ws.resif.fr")


# Pandas version check
from pkg_resources import parse_version
if parse_version(pd.__version__) != parse_version(u'0.18.0'):
    raise RuntimeError('Invalid pandas version')


utc = pytz.utc
END_TIME = UTCDateTime(2016, 1, 1)

RESIF_sta_names = list(['NET', 'STA', 'LOCID', 'CHA', 'LAT', 'LON', 'ELEV',
                        'DEP', 'AZ', 'DIP', 'INSTRUM', 'SCALE', 'SFREQ',
                        'SUNIT', 'SAMPRATE','START', 'END'])


class SeismicStation(object):
    
    def __init__(self, sta_lines, sta_col_names):

        nr, nc = sta_lines.shape

        ista = sta_col_names.index('STA')
        ilat = sta_col_names.index('LAT')
        ilon = sta_col_names.index('LON')
        istart = sta_col_names.index('START')
        iend = sta_col_names.index('END')
        iinstrum = sta_col_names.index('INSTRUM')
        iloc = sta_col_names.index('LOCID')

        d_format = '%Y-%m-%dT%H:%M:%S.%f'

        self.sta = np.unique(sta_lines[:, ista])[0]

        lats = np.empty(nr, dtype=float)
        lons = np.empty(nr, dtype=float)
        start_times = np.empty(nr, dtype=object)
        end_times = np.empty(nr, dtype=object)
        durations = np.empty(nr, dtype=object)
        instrums = np.empty(nr, dtype=object)
        locs = np.empty(nr, dtype=object)

        for i in xrange(nr):
            line = sta_lines[i]
            lats[i] = float(line[ilat])
            lons[i] = float(line[ilon])
            start_times[i] = UTCDateTime(line[istart])
            try:
                end_times[i] = UTCDateTime(line[iend])
            except TypeError:
                # case of missing end time
                end_times[i] = END_TIME
            if end_times[i] > END_TIME:
                end_times[i] = END_TIME
            durations[i] = end_times[i] - start_times[i]
            instrums[i] = line[iinstrum]
            locs[i] = line[iloc]

        self.lat = np.mean(lats)
        self.lon = np.mean(lons)

        self.n_periods = len(start_times)
        time_indexes = np.argsort(start_times)
        self.start_times = start_times[time_indexes]
        self.end_times = end_times[time_indexes]
        self.durations = durations[time_indexes]
        self.lats = lats[time_indexes]
        self.lons = lons[time_indexes]
        self.instrums = instrums[time_indexes]
        self.locs = locs[time_indexes]

        self.total_duration = np.sum(self.durations)

    def __str__(self):

        message = "%s \t (%7.4f, %9.4f)" % (self.sta, self.lat, self.lon)
        try:
            message = message + "\nZone %d (near %s)" % (self.zone, self.town)
        except AttributeError:
            pass
        days = int(self.total_duration / 86400)
        seconds = self.total_duration - days * 86400
        message = message + "\nTotal duration : %d days, %f s" % \
            (days , seconds)
        for i in xrange(self.n_periods):
            message = message + "\n %2d : %s --> %s (%7.4f, %9.4f) %s locid=%s" %\
                      (i+1, self.start_times[i].isoformat(),
                       self.end_times[i].isoformat(),
                       self.lats[i], self.lons[i], self.instrums[i], self.locs[i])
        message = message + "\n"
        return message


def get_RESIF_info(request, filename):
    urllib.urlretrieve(request, filename)

def read_sta_file(filename):
    
    pd_sta = pd.read_csv(filename, sep='|', header=0, names=RESIF_sta_names)
    short_list = list(['NET', 'STA', 'LOCID', 'LAT', 'LON', 'INSTRUM', 'SAMPRATE', 'START', 'END'])
    values =  pd_sta[short_list].values

    values_unique = np.vstack({tuple(row) for row in values})

    nr, nc = values_unique.shape

    ista = short_list.index('STA')
    ilat = short_list.index('LAT')
    ilon = short_list.index('LON')
    irat = short_list.index('SAMPRATE')
    istart = short_list.index('START')
    iend = short_list.index('END')

    for v in values_unique:
        v[ista] = v[ista].strip()
        
    return values_unique, short_list

def create_station_objects(X, names_list):

    ista = names_list.index('STA')
    stations = X[:, ista]
    unique_stations = np.unique(stations)

    nsta = len(unique_stations)
    
    stadict = {}
    for sta in unique_stations:
        lines = X[X[:, ista]==sta, :]
        stadict[sta] = SeismicStation(lines, names_list) 

    return stadict

def read_MC3_dump_file(filename):

    locale.setlocale(locale.LC_NUMERIC, 'fr_FR')

    data_frame = pd.read_table(
        filename,
        delimiter=';', encoding='ascii', skiprows=0,
        na_values=None, comment='#', header=None,
        thousands=None, skipinitialspace=True, mangle_dupe_cols=False
    )


    # Delete columns: 2, 3, 4, and 5
    columns = [2, 3, 4, 5]
    data_frame.drop(columns, axis=1, inplace=True)

    # Delete column 7
    data_frame.drop(7, axis=1, inplace=True)

    # Delete column 9
    data_frame.drop(9, axis=1, inplace=True)

    # Rename 0 to "WINDOW_START"
    data_frame.rename(columns={0: 'WINDOW_START'}, inplace=True)

    # Rename 1 to "WINDOW_LENGTH"
    data_frame.rename(columns={1: 'WINDOW_LENGTH'}, inplace=True)

    # Rename 6 to "EVENT_TYPE"
    data_frame.rename(columns={6: 'EVENT_TYPE'}, inplace=True)

    # Rename 8 to "LOCATED"
    data_frame.rename(columns={8: 'LOCATED'}, inplace=True)

    # Rename 10 to "ANALYST"
    data_frame.rename(columns={10: 'ANALYST'}, inplace=True)

    # Convert WINDOW_LENGTH to float
    data_frame[u'WINDOW_LENGTH'] = data_frame[u'WINDOW_LENGTH'].apply(locale.atof)

    # Convert WINDOW_START to datetime
    # data_frame[u'WINDOW_START'] = to_datetime(data_frame[u'WINDOW_START'])

    return data_frame


def get_webservice_data(net, sta, cha, starttime, endtime):
    url = 'http://eida.ipgp.fr/fdsnws/dataselect/1/query?'
    url = url + 'network=%s' % net
    url = url + '&station=%s' % sta
    url = url + '&channel=%s' % cha
    url = url + '&starttime=%s' % starttime
    url = url + '&endtime=%s' % endtime
    url = url + '&nodata=404'

    f_, fname = tempfile.mkstemp()
    os.close(f_)
    urllib.urlretrieve(url, fname)
    
    return fname
    # st = client.get_waveforms(net, sta, '??', cha, starttime, endtime)
    # return st


def get_webservice_metadata(net, fname):
    url = 'http://eida.ipgp.fr/fdsnws/station/1/query?'
    url = url + 'network=%s' % net
    url = url + '&level=response'
    url = url + '&format=xml' 
    url = url + '&nodata=404'

    urllib.urlretrieve(url, fname)

def get_data_from_catalog_entry(starttime, window_length, net, sta, cha, inv):

    # calculate the window length needed to do proper tapering / filtering
    # before deconvolution (add 20% to the length, 10% before and after)

    pad_length = max(10., window_length * 1.2)

    s_time = starttime - pad_length
    e_time = starttime + window_length + pad_length

    full_len = e_time - s_time

    pre_filt = [0.005, 0.01, 35., 45.]

    fname = get_webservice_data(net, sta, cha, s_time.isoformat(), e_time.isoformat())

    try:
        st = read(fname)

    except TypeError:
        _f = open(fname, 'r')
        lines = _f.readlines()
        _f.close()
        os.unlink(fname)
        print lines
        return None
        
    os.unlink(fname)
    st.detrend()
    st.attach_response(inv)
    st.remove_response(pre_filt=pre_filt)
    st = st.slice(starttime, starttime + window_length)

    return st

def get_catalog_entry(catalog_df, i):
    ev = catalog_df[i:i+1]
    starttime = ev['WINDOW_START'].values[0]
    window_length = ev['WINDOW_LENGTH'].values[0]
    event_type = ev['EVENT_TYPE'].values[0]
    analyst = ev['ANALYST'].values[0]

    starttime_obspy = UTCDateTime(starttime)

    return starttime_obspy, window_length, event_type, analyst
