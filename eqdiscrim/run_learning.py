import eqdiscrim_io as io
import attributes as att
import pandas as pd
import numpy as np
import pickle
import time
from obspy import read_inventory

do_get_metadata = False
do_plot_examples = False
do_calc_attributes = False

def get_data_and_attributes(catalog_df):
    n_events = len(catalog_df)
    for i in xrange(n_events):
        starttime, window_length, event_type, analyst = io.get_catalog_entry(catalog_df, i)
        try:
            st = io.get_data_from_catalog_entry(starttime, window_length, 'PF', 'BON', '???', inv)
            attributes, att_names = att.get_all_single_station_attributes(st)
            if i  == 0:
                df = pd.DataFrame(attributes, columns=att_names)
            else:
                df_tmp = pd.DataFrame(attributes, columns=att_names)
                df = df.append(df_tmp, ignore_index=True)
        except IOError:
            print 'Problem at %d' % i
            nan = np.ones((1, len(att_names)))*np.nan
            df_tmp = pd.DataFrame(nan, columns=att_names)
            df = df.append(df_tmp, ignore_index=True)
            continue
    df_X = catalog_df.join(df)
    return df_X
 

# first get metadata for all the stations
response_fname = 'PF_response.xml'
if do_get_metadata:
    io.get_webservice_metadata('PF', response_fname)
inv = read_inventory(response_fname)

# read catalog
catalog_fname = '../static_catalogs/MC3_dump_OVPF_2014.csv'
catalog_df = io.read_MC3_dump_file(catalog_fname)
som_df = catalog_df.query('EVENT_TYPE == "Sommital"')
eff_df = catalog_df.query('EVENT_TYPE == "Effondrement"')
loc_df = catalog_df.query('EVENT_TYPE == "Local"')
son_df = catalog_df.query('EVENT_TYPE == "Onde sonore"')
tel_df = catalog_df.query('EVENT_TYPE == "Teleseisme"')
phT_df = catalog_df.query('EVENT_TYPE == "Phase T"')
# print catalog_df
print som_df, eff_df, loc_df, son_df, tel_df, phT_df 
print catalog_df['EVENT_TYPE'].value_counts()


#  get and plot selected events
if do_plot_examples:
    sel_events = [
        [558, 'example_som.png'],
        [833, 'example_eff.png'],
        [262, 'example_loc.png'],
        [727, 'example_son.png'],
        [4065, 'example_tel.png'],
        [502, 'example_phT.png'],
    ]

    # get data and plot
    start = time.time()
    for ev in sel_events:
        starttime, window_length, event_type, analyst = io.get_catalog_entry(catalog_df, ev[0])
        # get the data (deconvolved)
        st = io.get_data_from_catalog_entry(starttime, window_length, 'PF', 'BON', '???', inv)
        print(st.__str__(extended=True))
        st.plot(outfile=ev[1])
    end = time.time()
    print "Time taken to get, deconvolve and plot selected data %.2f" % (end-start)

# get data and calculate attributes
df_X_fname = 'X_dataframe.dat'
test_df = catalog_df
if do_calc_attributes:
    start = time.time()
    df_X = get_data_and_attributes(test_df)
    end = time.time()
    f_ = open(df_X_fname, 'w')
    pickle.dump(df_X, f_)
    f_.close()
    print "Time taken to get and process %d events : %.2f" % (len(df_X), end - start)
else:
    start = time.time()
    f_ = open(df_X_fname, 'r')
    df_X = pickle.load(f_)
    f_.close()
    end = time.time()
    print "Time taken to read table for %d events : %.2f" % (len(df_X), end - start)
    
# do learning
print df_X.tail()
