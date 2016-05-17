import os
import numpy as np
from pickle import dump
from time import time
from cat_io import read_ovpf_cat
from seis_proc import extract_events, get_events_intersection
from seis_proc import extract_all_attributs, extract_types


do_extract_seis = True
do_extract_X = True
do_extract_Y = True
do_extract_VF = False

# do_extract_seis = True and do_extract_VF = True => extract seismograms for Vf only
# from all of the catalog, else extract seismogram for the whole catalogue.
 
X_filename = 'X.dat'
Y_filename = 'Y.dat'

# 2250 som, 727 eff in all the catalogue
# 2926 som, 256 eff for vf

# read catalog
t0 = time()
cat = read_ovpf_cat('MC3_dump.csv')

if do_extract_VF:
    # extract times and durations for VF events only
    cat_vf = cat[:, :][cat[:, -1] == 'VF']
else:
    cat_vf = cat

stimes_vf = cat_vf[:, 0]
dur_vf = cat_vf[:, 1]
type_vf = cat_vf[:, 2]
t1 = time()
print "Extracted VF catalog in %.2f s"%(t1-t0)

# extract the corresponding seismomgrams
if do_extract_seis:
    print "Extracting seismograms..."
    t0 = time()
    extract_events(stimes_vf, dur_vf, remove_response=True)
    t1 = time()
    print "Extracted seismograms in %.2f s"%(t1-t0)

# get list of partial filenames for events common to all (both) stations
t0 = time()
common_fileglob = get_events_intersection()
common_fileglob = list(common_fileglob)
nev = len(common_fileglob) 
t1 = time()
print "Extracted list of events recorded at both stations in %.2f s"%(t1-t0)

if do_extract_X:
    t0 = time()
    # extract attributes
    X, att_names = extract_all_attributs(common_fileglob)
    t1 = time()
    print X.shape, att_names
    print "Extracted attributes in %.2f s"%(t1-t0)
    # dump X to file
    f_ = open(X_filename, 'w')
    dump((X, att_names), f_)
    f_.close()
    
if do_extract_Y:
    t0 = time()
    # extract the types
    Y, Ydict = extract_types(cat_vf, common_fileglob)
    t1 = time()
    print len(Y)
    print Ydict
    print "Extracted types in %.2f s"%(t1-t0)
    # dump Y to file
    f_ = open(Y_filename, 'w')
    dump((Y, Ydict), f_)
    f_.close()

