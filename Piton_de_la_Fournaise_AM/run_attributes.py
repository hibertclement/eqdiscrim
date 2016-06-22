import eqdiscrim_io as io
import attributes as att
import pandas as pd
import numpy as np
import pickle
import time
import os
from obspy import read_inventory, UTCDateTime

do_read_dump = False
do_get_metadata = False
do_calc_attributes = True

# catalog name
catalog_fname = 'MC3_dump_2009_2016.csv'
catalog_df_fname = 'df_MC3_dump_2009_2016.dat'

event_types = ["Local", "Profond", "Regional", "Teleseisme", "Onde sonore", "Phase T", "Sommital", "Effondrement"]

station_names = ["RVL", "FLR", "BOR", "BON", "SNE", "FJS", "CSS", "GPS", "GPN", "FOR"]
max_events_per_file = 10
att_dir = "Attributes"

if not os.path.exists(att_dir):
    os.mkdir(att_dir)

def get_data_and_attributes(catalog_df, staname, start_i=0, n_max=None, obs='OVPF'):
    if n_max is None:
        n_events = len(catalog_df)
    else:
        n_events = n_max
    for i in xrange(n_events):
        ii = i + start_i
        try:
            starttime, window_length, event_type, analyst = \
            io.get_catalog_entry(catalog_df, ii)
            print staname, ii, starttime.isoformat()
            st = io.get_data_from_catalog_entry(starttime, window_length, 'PF',
                                                staname, '???', inv, obs=obs)
            attributes, att_names = att.get_all_single_station_attributes(st)
            if i  == 0:
                df = pd.DataFrame(attributes, columns=att_names, index=[ii])
            else:
                df_tmp = pd.DataFrame(attributes, columns=att_names, index=[ii])
                df = df.append(df_tmp, ignore_index=False)
        except:
            print 'Problem at %d - Setting all attributes to NaN' % ii
            nan = np.ones((1, len(att_names)))*np.nan
            df_tmp = pd.DataFrame(nan, columns=att_names)
            df = df.append(df_tmp, ignore_index=False)
            continue
    df_X = catalog_df.iloc[start_i : start_i + n_events].join(df)
    return df_X
 
# first get metadata for all the stations
response_fname = 'PF_response.xml'
if do_get_metadata:
    print("Getting station xml for PF network")
    io.get_webservice_metadata('PF', response_fname)
inv = read_inventory(response_fname)

# read catalog
if do_read_dump:
    print("Reading OVPF catalog and writing dataframe")
    catalog_df = io.read_MC3_dump_file(catalog_fname)
    f_ = open(catalog_df_fname, 'w')
    pickle.dump(catalog_df, f_)
    f_.close()
else:
    print("Reading OVPF catalog dataframe")
    f_ = open(catalog_df_fname, 'r')
    catalog_df = pickle.load(f_)
    f_.close()
print catalog_df['EVENT_TYPE'].value_counts()

# create dictionaries according to types
event_type_df_dict = {}
for ev_type in event_types:
    event_type_df_dict[ev_type] = catalog_df[catalog_df['EVENT_TYPE']==ev_type]
    print event_type_df_dict[ev_type].head()

n_events = len(catalog_df)

# get data and calculate attributes
if do_calc_attributes:
    for s in station_names: 
        i_start = 0
        while i_start < n_events:
            n_max = min(max_events_per_file, n_events - i_start)
            df_X_fname = os.path.join(att_dir, 'X_%s_%05d_dataframe.dat' % (s, i_start))
            if not os.path.exists(df_X_fname):
                df_X = get_data_and_attributes(catalog_df, s, i_start, n_max, 'OVPF')
                f_ = open(df_X_fname, 'w')
                pickle.dump(df_X, f_)
                f_.close()
                print "Wrote %s" % df_X_fname
            else:
                print "%s exists - moving on to next set" % df_X_fname
            i_start += max_events_per_file
