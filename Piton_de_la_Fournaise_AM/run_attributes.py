import eqdiscrim_io as io
import attributes as att
import pandas as pd
import numpy as np
import pickle
import time
import os
from obspy import read_inventory, UTCDateTime

do_get_metadata = False
do_calc_attributes = True

# catalog name
catalog_fname = 'MC3_dump_2012_2016.csv'

station_names = ["RVL", "FLR", "BOR", "BON", "SNE", "FJS", "CSS", "GPS", "GPN", "FOR"]
max_events_per_file = 10
#station_names = ["BON"]
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
print("Reading OVPF catalog")
catalog_df = io.read_MC3_dump_file(catalog_fname)
print catalog_df['EVENT_TYPE'].value_counts()

n_events = len(catalog_df)

# get data and calculate attributes
if do_calc_attributes:
    for s in station_names: 
        i_start = 0
        while i_start < n_events:
            n_max = min(max_events_per_file, n_events - i_start)
            df_X_fname = os.path.join(att_dir, 'X_%s_%05d_dataframe.dat' % (s, i_start))
            if not os.path.exists(df_X_fname):
                start = time.time()
                df_X = get_data_and_attributes(catalog_df, s, i_start, n_max, 'OVPF')
                print df_X
                end = time.time()
                f_ = open(df_X_fname, 'w')
                pickle.dump(df_X, f_)
                f_.close()
                print "Time taken to get and process %d events starting at %d at %s : %.2f" % \
                  (n_max, i_start, s, end - start)
            else:
                print "%s exists - moving on to next set" % df_X_fname
            i_start += max_events_per_file
