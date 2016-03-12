import os
import data_io as io
from obspy.core import read, UTCDateTime

datadir = 'test_data'

fname = os.path.join(datadir, 'catalog_test_version.xls')
data_regex = os.path.join(datadir, 'YA*')

cat = io.read_catalog(fname)
st_list = io.read_and_cut_events(cat, data_regex)

# process and write the events from the CH catalog
i=0
for st in st_list:
    i += 1
    #st.plot()
    for tr in st:
        tr.detrend('linear')
        fname = os.path.join(datadir, 'events',
                              "event_%02d_%s_%s.SAC" % (i, tr.get_id(),
                                            tr.stats.starttime.isoformat()))
        tr.write(fname, format='SAC')

# prepare the test_data for polarisation tests

fname = os.path.join(datadir, 'IU*.sac')
st = read(fname, starttime = UTCDateTime('2002-11-03T22:19:28.000000Z'),
          endtime=UTCDateTime('2002-11-03T22:27:28.000000Z'))

st.plot()
for tr in st:
    fname = os.path.join(datadir, "pol_%s.sac" % (tr.get_id()))
    tr.write(fname, format='SAC')
