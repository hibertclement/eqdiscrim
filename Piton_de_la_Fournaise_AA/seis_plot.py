# -*- coding: utf-8 -*-
import numpy as np
from os.path import join
from obspy.core import read
import matplotlib.pyplot as plt
from cat_io import read_ovpf_cat
from seis_proc import extract_types, stations, edir


# number of random events to plot
N_EVENTS = 10

# read cat
cat = read_ovpf_cat('MC3_dump.csv')
cat_vf = cat[:, :][cat[:, -1] == 'VF']

# get the types of the events
Ynames = cat_vf[:, 2]
durations = cat_vf[:, 1]
Ynames_unique = np.unique(Ynames)
ntypes = len(Ynames_unique)
nsta = len(stations)

# make the empty strucutres to keep the seismograms
data_list = []
for sta in stations:
    data_list.append(np.empty((N_EVENTS, ntypes), dtype=object))

# keep the number of files read and the maximum duration
atype_n = np.zeros(ntypes)
atype_dur = np.zeros(ntypes)

# Read in the data. Be careful that the seismograms may not exist for all the
# events in the catalogue !!

# for each type :
for itype in xrange(ntypes):
    atype = Ynames_unique[itype]
    print 'Extracting %s'%atype
    stimes = cat_vf[:, 0][Ynames == atype]
    dur = cat_vf[:, 1][Ynames == atype]
    rand_idx = np.random.permutation(len(stimes))

    # for each index (random order)
    for ista in xrange(nsta):
        n_ok = 0
        for idx in rand_idx:
            # make filename
            sta = stations[ista]
            fname = join(edir, "%s_*.%s.*MSEED"%(stimes[idx].isoformat(),
                         sta.split('.')[0]))

            try:
                st = read(fname)
                data_list[ista][n_ok, itype] = st
                print 'Read %s.'%fname
                if dur[idx] > atype_dur[itype]:
                    atype_dur[itype] = dur[idx]
                n_ok += 1
                if n_ok == N_EVENTS:
                    break
            except:
                #print 'File %s does not exist, moving to next file.'%fname
                pass

    # have found all files for this type
    # save their number
    atype_n[itype] = n_ok


# do plot
for itype in xrange(ntypes):
    atype = Ynames_unique[itype]
    print 'Plotting %s'%atype
    nev = int(atype_n[itype])
    print nev, nsta
    if nev == 0:
        continue
    plt.clf()
    fig, axes = plt.subplots(nev, nsta)
    for iev in xrange(nev):
        for ista in xrange(nsta):
            if nev > 1:
                plt.sca(axes[iev, ista])
            else:
                plt.sca(axes[ista])
            st.taper(max_percentage=0.05, type='cosine')
            st = data_list[ista][iev, itype]
            st.filter('bandpass', freqmin=4.0, freqmax=10.0)
            dt = st[0].stats.delta
            npts = st[0].stats.npts
            e_time = (npts-1) * dt
            times = np.linspace(0.0, e_time, num=npts)
            plt.plot(times, st[0].data, 'b', label=st[0].stats.station)
            plt.xlim(0, atype_dur[itype])
            if iev == nev-1:
                plt.xlabel('Time (s)')
                plt.ylabel('Velocity (um/s)')

    fig.savefig('%s_%d.png'%(atype, N_EVENTS))
  
