import eqdiscrim_io as io
import attributes as att
import pandas as pd
import numpy as np
import argparse
import os
from obspy import read_inventory, read
from obspy.io.xseed import Parser


# -----------------------------------
# FUNCTIONS
# -----------------------------------

def get_data_and_attributes(cfg, inv, catalog_df, staname, indexes,
                            obs='OVPF'):
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

            if cfg.do_use_saved_data:
                st_fname = os.path.join(
                    cfg.data_dir, "%d_PF.%s.*MSEED" % (index, staname))
                try:
                    st = read(st_fname)
                except:
                    st = None

            else:
                if staname == 'BOR':
                    parser = Parser(cfg.BOR_response_fname)
                    st = io.get_waveform_data(starttime,
                                                        window_length,
                                                        'PF', staname,
                                                        '??Z', parser,
                                                        obs=obs,
                                                        simulate=True)
                else:
                    st = io.get_waveform_data(starttime,
                                                        window_length,
                                                        'PF', staname, '??Z',
                                                        inv, obs=obs)
                if cfg.do_save_data and st is not None:
                    for tr in st:
                        tr_fname = os.path.join(cfg.data_dir,
                                                "%d_%s.MSEED" % (index,
                                                                 tr.get_id()))
                        tr.write(tr_fname, format='MSEED')

            # actually get the attributes
            try:
                st.detrend()
            except AttributeError:
                raise ValueError
            attributes, att_names =\
                att.get_all_single_station_attributes(st)

            # create the data-frame with the attributes (using the same indexes
            # as those in the catalog)
            if i == 0:
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
            try:
                att_names = df_att.columns.values
            except UnboundLocalError:
                att_names = att.att_names_single_station_1D
            nan = np.ones((1, len(att_names))) * np.nan
            df_att_tmp = pd.DataFrame(nan, columns=att_names, index=[index])
            try:
                df_att = df_att.append(df_att_tmp, ignore_index=False)
            except UnboundLocalError:
                df_att = df_att_tmp
            continue

    # join the attributes onto the portion of the catalog data-frame
    # we are working with and return it
    df_X = catalog_df.ix[indexes].join(df_att)
    return df_X


def calc_and_write_attributes(cfg, inv, df_samp, ev_type, staname, att_dir,
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
            df_X = get_data_and_attributes(cfg, inv, df_samp, staname,
                                           indexes[i_start:i_start + n_max],
                                           'OVPF')
            # save resulting data-frame to file
            io.dump(df_X, df_X_fname)
            print "Wrote %s" % df_X_fname
        else:
            # file exists - do nothing
            print "%s exists - moving on to next set" % df_X_fname

        # go onto next batch
        i_start += n_max


def run_attributes(args):

    cfg = io.Config(args.config_file)

    # make output directories if they do not exist
    if not os.path.exists(cfg.att_dir):
        os.mkdir(cfg.att_dir)
    if cfg.do_save_data and not os.path.exists(cfg.data_dir):
        os.mkdir(cfg.data_dir)

    # get metadata for all the stations
    if cfg.do_get_metadata:
        print("Getting station xml for PF network")
        io.get_webservice_metadata('PF', cfg.response_fname)
    inv = read_inventory(cfg.response_fname)

    # read catalog
    if cfg.do_read_dump:
        # read the dump file
        print("\nReading OVPF catalog and writing dataframe")
        catalog_df = io.read_MC3_dump_file(cfg.catalog_fname)
        io.dump(catalog_df, cfg.catalog_df_fname)
    else:
        # read the full dataframe file
        print("\nReading OVPF catalog dataframe")
        io.load(cfg.catalog_df_fname)
    print "Full catalog :"
    print catalog_df['EVENT_TYPE'].value_counts()

    # sample the database
    if cfg.do_sample_database:
        # Re-sample the database
        print("\nResampling with maximum number of events per type = %d"
              % cfg.max_events_per_type)
        event_df_list = []
        for ev_type in cfg.event_types:
            event_df = catalog_df[catalog_df['EVENT_TYPE'] == ev_type]
            n_events = len(event_df)
            if n_events > cfg.max_events_per_type:
                n_events = cfg.max_events_per_type
                df_samp = event_df.sample(n=n_events)
            else:
                df_samp = event_df.copy()
            event_df_list.append(df_samp)
        # ensure permanence of samples by writing to file
        sampled_df = pd.concat(event_df_list)
        io.dump(sampled_df, cfg.catalog_df_samp_fname)
    else:
        # read the sampled database
        print("\nReading the sampled dataframe")
        sampled_df = io.load(cfg.catalog_df_samp_fname)
    print "Sampled catalog :"
    print sampled_df['EVENT_TYPE'].value_counts()

    # remove problematic events
    try:
        sampled_df.drop(3514, axis=0, inplace=True)
    except ValueError:
        pass

    # create dictionaries of sampled sub-databases according to types
    event_type_df_dict = {}
    for ev_type in cfg.event_types:
        event_type_df_dict[ev_type] = sampled_df[sampled_df['EVENT_TYPE'] ==
                                                 ev_type]

    # calculate the attributes
    if cfg.do_calc_attributes:
        # loop over event types
        for ev_type in cfg.event_types:
            df = event_type_df_dict[ev_type]
            for staname in cfg.station_names:
                calc_and_write_attributes(cfg, inv, df, ev_type, staname,
                                          cfg.att_dir, cfg.max_events_per_file)


if __name__ == '__main__':

    # set up parser
    parser = argparse.ArgumentParser(
        description='Launch attribute calculation for classifier training')
    parser.add_argument('config_file', help='eqdiscrim configuration file')

    # parse input
    args = parser.parse_args()

    # run program
    run_attributes(args)
