# -*- coding: utf-8 -*-

import os
import pickle
import locale
import pytz
import urllib
import tempfile 
import pandas as pd
import numpy as np
import itertools as it
import ConfigParser
import base64
from obspy import UTCDateTime, read
from obspy.clients.fdsn import Client
from datetime import timedelta
from obspy.clients.arclink import Client


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

class Config(object):

    def __init__(self, fname):
        config = ConfigParser.ConfigParser()
        config.read(fname)

        # Catalogs
        self.catalog_fname = config.get('Catalogs', 'catalog_fname')
        self.catalog_df_fname = config.get('Catalogs', 'catalog_df_fname')
        self.catalog_df_samp_fname = config.get('Catalogs', 'catalog_df_samp_fname')
        self.do_read_dump = config.getboolean('Catalogs', 'do_read_dump')
        self.do_sample_database = config.getboolean('Catalogs', 'do_sample_database')

        # Metadata
        self.response_fname = config.get('Metadata', 'response_fname')
        self.BOR_response_fname = config.get('Metadata', 'BOR_response_fname')
        self.do_get_metadata = config.getboolean('Metadata', 'do_get_metadata')

        # Attributes
        self.att_dir = config.get('Attributes', 'att_dir')
        self.max_events_per_file = config.getint('Attributes', 'max_events_per_file')
        self.do_calc_attributes = config.getboolean('Attributes', 'do_calc_attributes')

        # Data
        self.data_dir = config.get('Data', 'data_dir')
        self.do_save_data = config.getboolean('Data', 'do_save_data')
        self.do_use_saved_data = config.getboolean('Data', 'do_use_saved_data')

        # Classes
        self.max_events_per_type = config.getint('Classes', 'max_events_per_type')
        self.event_types = self.parse_list_(config.get('Classes', 'event_types'))

        # Stations
        self.station_names = self.parse_list_(config.get('Stations', 'station_names'))

        # Figures
        self.figdir = config.get('Figures', 'figdir')

        # Learning
        self.do_learning_curve = config.getboolean('Learning', 'do_learning_curve')
        self.max_events = config.getint('Learning', 'max_events')
        self.n_best_atts = config.getint('Learning', 'n_best_atts')
        self.best_atts_fname = config.get('Learning', 'best_atts_fname')
        self.clf_fname = config.get('Learning', 'clf_fname')

    def parse_list_(self, list_as_string):
        words = list_as_string.split(',')
        ret_list = []
        for w in words:
            ret_list.append(w.strip())
        return ret_list

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

def get_OVPF_MC3_dump_file(s_time, e_time, filename, evtype=None):

    username = "ferraz"
    password = "inizzarref"

    starttime = s_time
    endtime = starttime + 3600

    all_data = ""
    while endtime < e_time:

        y1 = starttime.year
        m1 = starttime.month
        d1 = starttime.day

        y2 = endtime.year
        m2 = endtime.month
        d2 = endtime.day


        url = "http://pitondescalumets.ipgp.fr/cgi-bin/mc3.pl?"
        url = url + "y1=%d&m1=%d&d1=%d" % (y1, m1, d1)
        url = url + "y2=%d&m2=%d&d2=%d" % (y2, m2, d2)
        url = url + "&dump=bul"
    
        if evtype is not None:
            url = url + "type=%s" % evtype

        thepage = deal_with_authentication(url, username, password)
        all_data += thepage

        starttime = endtime 
        endtime = starttime + 3600
        

    f_ = open(filename, 'w')
    f_.write("%s" % all_data)
    f_.close()

    return all_data

def deal_with_authentication(theurl, username, password):

    import urllib2
    import sys
    import re
    import base64
    from urlparse import urlparse

    req = urllib2.Request(theurl)
    try:
        handle = urllib2.urlopen(req)
    except IOError, e:
        # here we *want* to fail
        pass
    else:
        # If we don't fail then the page isn't protected
        print "This page isn't protected by authentication."
        sys.exit(1)

    if not hasattr(e, 'code') or e.code != 401:
        # we got an error - but not a 401 error
        print "This page isn't protected by authentication."
        print 'But we failed for another reason.'
        sys.exit(1)

    authline = e.headers['www-authenticate']
    # this gets the www-authenticate line from the headers
    # which has the authentication scheme and realm in it

    authobj = re.compile(
        r'''(?:\s*www-authenticate\s*:)?\s*(\w*)\s+realm=['"]([^'"]+)['"]''',
        re.IGNORECASE)
    # this regular expression is used to extract scheme and realm
    matchobj = authobj.match(authline)

    if not matchobj:
        # if the authline isn't matched by the regular expression
        # then something is wrong
        print 'The authentication header is badly formed.'
        print authline
        sys.exit(1)

    scheme = matchobj.group(1)
    realm = matchobj.group(2)
    # here we've extracted the scheme
    # and the realm from the header
    if scheme.lower() != 'basic':
        print 'This example only works with BASIC authentication.'
        sys.exit(1)

    base64string = base64.encodestring(
                '%s:%s' % (username, password))[:-1]
    authheader =  "Basic %s" % base64string
    req.add_header("Authorization", authheader)
    try:
        handle = urllib2.urlopen(req)
    except IOError, e:
        # here we shouldn't fail if the username/password is right
        print "It looks like the username or password is wrong."
        sys.exit(1)

    thepage = handle.read()

    return thepage

def read_MC3_dump_file(filename):

    if os.environ["LANG"] is "r_FR.UTF-8":
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
    if os.environ["LANG"] is "r_FR.UTF-8":
        data_frame[u'WINDOW_LENGTH'] = data_frame[u'WINDOW_LENGTH'].apply(locale.atof)

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

def get_OVPF_arclink_data(net, sta, locid, cha, starttime, endtime):

    # serveur de données OVPF
    # pitonmanuel 195.83.188.22
    client = Client(host="195.83.188.22", port="18001", user="sysop",
                    institution="OVPF")

    try:
        #print client.get_inventory(net, sta, locid, cha, starttime, endtime).keys()
        st = client.get_waveforms(net, sta, locid, cha, starttime, endtime)
        return st
    except:
        return None


def get_webservice_metadata(net, fname):
    url = 'http://eida.ipgp.fr/fdsnws/station/1/query?'
    url = url + 'network=%s' % net
    url = url + '&level=response'
    url = url + '&format=xml' 
    url = url + '&nodata=404'

    urllib.urlretrieve(url, fname)

def get_data_from_catalog_entry(starttime, window_length, net, sta, cha, inv,
                                obs='OVPF', simulate=False):

    # calculate the window length needed to do proper tapering / filtering
    # before deconvolution (add 20% to the length, 10% before and after)

    pad_length = max(60., window_length * 0.2)

    s_time = starttime - pad_length
    e_time = starttime + window_length + pad_length

    full_len = e_time - s_time

    pre_filt = [0.005, 0.01, 35., 45.]

    if obs is 'OVPF':
        st = get_OVPF_arclink_data(net, sta, "*", cha, s_time, e_time)
        if st is None:
            return None
    else:
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
    if simulate:
        # inv actually contains a parser object that has read a dataless seed file
        st.simulate(seedresp={'filename': inv, 'units': "VEL"})
    else:
        # inv contains a stationxml inventory complete with responses
        st.attach_response(inv)
        st.remove_response(pre_filt=pre_filt)
    st = st.slice(starttime, starttime + window_length)
    st.detrend()

    return st

def get_catalog_entry(catalog_df, index):
    ev = catalog_df.ix[index]
    starttime = ev['WINDOW_START']
    window_length = ev['WINDOW_LENGTH']
    event_type = ev['EVENT_TYPE']
    analyst = ev['ANALYST']

    starttime_obspy = UTCDateTime(starttime)

    return starttime_obspy, window_length, event_type, analyst

def read_and_cat_dataframes(fnames):

    for fname in fnames:
        f_ = open(fname, 'r')
        X_df = pickle.load(f_)
        f_.close()
        if fname is fnames[0]:
            X_df_full = X_df
        else:
            X_df_full = X_df_full.append(X_df, ignore_index=False)

    return X_df_full

def get_station_combinations(station_names):
    comb_list = []
    n_sta = len(station_names)
    for i in xrange(n_sta-1):
        ii = i + 2
        for comb in it.combinations(station_names, ii):
            comb_list.append(comb)

    return comb_list

