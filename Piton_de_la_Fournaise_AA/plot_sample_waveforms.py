import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from os.path import join
from obspy.core import read
from seis_proc import stations, edir

N_EV = 5

# select N_EV events at random
fnames = glob(join(edir, '*.MSEED'))
otime_list = []
for fname in fnames:
    otime_list.append(fname.split('_')[0])
uniq_otimes = np.unique(np.array(otime_list))
random_events = np.random.permutation(len(uniq_otimes))[0:N_EV]

fig, axes = plt.subplots(5, 1)

# read the seismograms into two arrays according to station
for iev in xrange(N_EV):
    st = read("%s_*.MSEED"%uniq_otimes[random_events[iev]])
    # filter in a common band (above 1-10Hz)
    st.taper(max_percentage=0.05, type='cosine')
    st.filter('bandpass', freqmin=4.0, freqmax=10.0)
    # get stats
    # for our stations the sample rate and duration are the same
    dt = st[0].stats.delta
    npts = st[0].stats.npts
    e_time = (npts-1) * dt
    times = np.linspace(0.0, e_time, num=npts)
    # plot each event in a separate window and overlay the two stations
    plt.sca(axes[iev])
    plt.plot(times, st[0].data, 'b', label=st[0].stats.station)
    plt.plot(times, st[1].data, 'r', label=st[1].stats.station)
    plt.xlabel('Time (s)')
    plt.ylabel('Velocity (um/s)')
    plt.legend()

plt.show()
import itertools as it

P = list(it.permutations(range(3)))
print P
P = list(it.permutations(range(4)))
print P
