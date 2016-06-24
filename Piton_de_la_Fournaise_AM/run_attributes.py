import eqdiscrim_io as io
import attributes as att
import pandas as pd
import numpy as np
import pickle
import time
import os
from obspy import read_inventory, UTCDateTime, read
from obspy.io.xseed import Parser

# -----------------------------------
# SWITCHES
# -----------------------------------

do_read_dump = False
do_sample_database = False
do_get_metadata = False
do_calc_attributes = True
do_fake_attributes = False
do_save_data = True
do_use_saved_data = False

# -----------------------------------
# LOGISTICS
# -----------------------------------

catalog_fname = 'MC3_dump_2009_2016.csv'
catalog_df_fname = 'df_MC3_dump_2009_2016.dat'
catalog_df_samp_fname = 'df_MC3_dump_2009_2016_sampled.dat'
event_types = ["Local", "Profond", "Regional", "Teleseisme", "Onde sonore",
               "Phase T", "Sommital", "Effondrement", "Indetermine"]
station_names = ["RVL", "BOR", "PRO", "CIL", "NSR", "NTR", "FJS", "FLR", "SNE", "BON", "BOR", "GPS", "TTR"]
max_events_per_file = 500
max_events_per_type = 2000
att_dir = "Attributes"
data_dir = "Data"
response_fname = 'PF_response.xml'
BOR_response_fname = 'OVPF-CP.BOR.dataless'

# -----------------------------------
# FUNCTIONS
# -----------------------------------

def get_fake_attributes(starttime):
    """
    Creates some fake attributes for testing purposes
    """
    att_names = ["Starttime", "Dummy_1", "Dummy_2", "Dummy_3", "Dummy_4"]
    att = np.empty((1, len(att_names)), dtype=object) 
    att[0, 0] = starttime.isoformat()
    att[0, 1:] = np.arange(4)[:]

    return att, att_names

def get_data_and_attributes(catalog_df, staname, indexes, obs='OVPF'):
    """
    Given a catalog data-frame, runs the requests for the data and
    calculates the attributes.
    """
    # Set the number of events in an attribute file (for convenience)
    n_events = len(indexes)

    # For each event in the catalog file
    for i in xrange(n_events):
        # get the number of the line to read
        index = indexes[i]

        try:
            # parse the catalog entry
            starttime, window_length, event_type, analyst =\
                io.get_catalog_entry(catalog_df, index)
            print event_type, staname, i, index, starttime.isoformat()

            # get the data and attributes
            if do_fake_attributes:
                attributes, att_names = get_fake_attributes(starttime)

            elif do_use_saved_data:
                st_fname = os.path.join(data_dir, "%d_PF.%s.*MSEED" % (index, staname))
                st = read(st_fname)

            else:
                if staname is 'BOR':
                        parser = Parser(BOR_response_fname)
                        st = io.get_data_from_catalog_entry(starttime, window_length,
                                                    'PF', staname, '??Z', parser,
                                                     obs=obs, simulate=True)
                else:
                    st = io.get_data_from_catalog_entry(starttime, window_length,
                                                    'PF', staname, '??Z', inv,
                                                     obs=obs)
                if do_save_data and st is not None:
                    for tr in st:
                        tr_fname = os.path.join(data_dir, "%d_%s.MSEED" % (index, tr.get_id()))
                        tr.write(tr_fname, format='MSEED')

                # actually get the attributes
                attributes, att_names =\
                    att.get_all_single_station_attributes(st)

            # create the data-frame with the attributes (using the same indexes
            # as those in the catalog)
            if i  == 0:
                df_att = pd.DataFrame(attributes, columns=att_names,
                                      index=[index])
            else:
                df_att_tmp = pd.DataFrame(attributes, columns=att_names,
                                      index=[index])
                df_att = df_att.append(df_att_tmp, ignore_index=False)

        except ValueError:
            # if there are problems, then set attributes to NaN
            print 'Problem at %d (%d)  - Setting all attributes to NaN' %\
                  (i, index)
            att_names = df_att.columns.values
            nan = np.ones((1, len(att_names))) * np.nan
            df_att_tmp = pd.DataFrame(nan, columns=att_names, index=[index])
            df_att = df_att.append(df_att_tmp, ignore_index=False)
            continue

    # join the attributes onto the portion of the catalog data-frame
    # we are working with and return it
    df_X = catalog_df.ix[indexes].join(df_att)
    return df_X
 
def calc_and_write_attributes(df_samp, ev_type, staname, att_dir,
                              max_events_per_file=None):
    """
    Outer function to calculate and write the attributes
    """

    n_events = len(df_samp)

    # save the indexes of the sampled catalog - they are needed later
    indexes = df_samp.index

    # batch calculations in groups of max_events_per_file - useful in case
    # runs crash for any reason during attribute calculation - program can
    # be relauched to complete any incomplete calculations
    i_start = 0
    while i_start < n_events:

        # get the number of events in this batch
        if max_events_per_file is not None:
            n_max = min(max_events_per_file, n_events - i_start)
        else:
            n_max = n_events

        # construct the filename for this batch
        df_X_fname = os.path.join(att_dir, 'X_%s_%s_%05d_dataframe.dat' %
                                           (ev_type, staname, i_start))

        # if the file does not already exist launch the process
        if not os.path.exists(df_X_fname):

            # get the data and attributes for this batch
            df_X = get_data_and_attributes(df_samp, staname,
                                           indexes[i_start : i_start + n_max],
                                           'OVPF')
            # save resulting data-frame to file
            f_ = open(df_X_fname, 'w')
            pickle.dump(df_X, f_)
            f_.close()
            print "Wrote %s" % df_X_fname
        else:
            # file exists - do nothing
            print "%s exists - moving on to next set" % df_X_fname

        # go onto next batch
        i_start += n_max


# CODE STARTS HERE

# make output directories if they do not exist
if not os.path.exists(att_dir):
    os.mkdir(att_dir)
if do_save_data and not os.path.exists(data_dir):
    os.mkdir(data_dir)

# get metadata for all the stations
if do_get_metadata:
    print("Getting station xml for PF network")
    io.get_webservice_metadata('PF', response_fname)
inv = read_inventory(response_fname)

# read catalog
if do_read_dump:
    # read the dump file
    print("\nReading OVPF catalog and writing dataframe")
    catalog_df = io.read_MC3_dump_file(catalog_fname)
    f_ = open(catalog_df_fname, 'w')
    pickle.dump(catalog_df, f_)
    f_.close()
else:
    # read the full dataframe file
    print("\nReading OVPF catalog dataframe")
    f_ = open(catalog_df_fname, 'r')
    catalog_df = pickle.load(f_)
    f_.close()
print "Full catalog :"
print catalog_df['EVENT_TYPE'].value_counts()

# sample the database
if do_sample_database:
    # Re-sample the database
    print("\nResampling the dataframe with maximum number of events per type = %d"
          % max_events_per_type)
    event_df_list = []
    for ev_type in event_types:
        event_df = catalog_df[catalog_df['EVENT_TYPE']==ev_type]
        n_events = len(event_df)
        if n_events > max_events_per_type:
            n_events = max_events_per_type
            df_samp = event_df.sample(n=max_events_per_type)
        else:
            df_samp = event_df.copy()
        event_df_list.append(df_samp)
    # ensure permanence of samples by writing to file
    sampled_df = pd.concat(event_df_list)
    f_ = open(catalog_df_samp_fname, 'w')
    pickle.dump(sampled_df, f_)
    f_.close()
else:
    # read the sampled database
    print("\nReading the sampled dataframe")
    f_ = open(catalog_df_samp_fname, 'r')
    sampled_df = pickle.load(f_)
    f_.close()
print "Sampled catalog :"
print sampled_df['EVENT_TYPE'].value_counts()


# create dictionaries of sampled sub-databases according to types
event_type_df_dict = {}
for ev_type in event_types:
    event_type_df_dict[ev_type] = sampled_df[sampled_df['EVENT_TYPE']==ev_type]

# calculate the attributes
if do_calc_attributes:
    # loop over event types
    for ev_type in event_types:
        df = event_type_df_dict[ev_type]
        for staname in station_names: 
            calc_and_write_attributes(df, ev_type, staname, att_dir,
                                      max_events_per_file)
