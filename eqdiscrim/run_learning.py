import eqdiscrim_io as io
import pandas as pd
from obspy import read_inventory, UTCDateTime
import time

do_get_metadata = True

# read catalog
catalog_fname = '../static_catalogs/MC3_dump_OVPF_2014.csv'
catalog_df = io.read_MC3_dump_file(catalog_fname)
som_df = catalog_df.query('EVENT_TYPE == "Sommital"')
eff_df = catalog_df.query('EVENT_TYPE == "Effondrement"')
loc_df = catalog_df.query('EVENT_TYPE == "Local"')
son_df = catalog_df.query('EVENT_TYPE == "Onde\ Sonore"')
tel_df = catalog_df.query('EVENT_TYPE == "Teleseisme"')
phT_df = catalog_df.query('EVENT_TYPE == "Phase T"')
print som_df, eff_df, loc_df, son_df, tel_df, phT_df 
print pd.value_counts(catalog_df['EVENT_TYPE'])


# first get metadata for all the stations
response_fname = 'PF_response.xml'
if do_get_metadata:
    io.get_webservice_metadata('PF', response_fname)
inv = read_inventory(response_fname)

sel_events = [
    [191, 'example_som.png'],
    [4053, 'example_eff.png'],
    [394, 'example_loc.png'],
    [3024, 'example_tel.png'],
    [502, 'example_phT.png'],
]

# go through the catalog 
for ev in sel_events:
    starttime, window_length, event_type, analyst = io.get_catalog_entry(catalog_df, ev[0])
    # get the data (deconvolved)
    start = time.time()
    st = io.get_data_from_catalog_entry(starttime, window_length, 'PF', 'BON', '???', inv)
    end = time.time()
    print "Time taken to get and deconvolve data %.2f" % (end-start)
    print(st.__str__(extended=True))
    st.plot(outfile=ev[1])

    # calculate attributes


# do learning


