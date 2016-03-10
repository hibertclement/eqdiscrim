import os
import data_io as io

datadir = 'test_data'

fname = os.path.join(datadir, 'catalog_test_version.xls')
data_regex = os.path.join(datadir, 'YA*')

cat = io.read_catalog(fname)
st_list = io.read_and_cut_events(cat, data_regex)

i=0
for st in st_list:
    i += 1
    st.plot()
    for tr in st:
        tr.detrend('linear')
        fname = os.path.join(datadir, 'events',
                             "event_%02d_%s_%s.SAC" % (i, tr.get_id(),
                                            tr.stats.starttime.isoformat()))
        tr.write(fname, format='SAC')
