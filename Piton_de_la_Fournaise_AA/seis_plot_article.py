# -*- coding: utf-8 -*-
import numpy as np
from os.path import join
from obspy.core import read
import matplotlib.pyplot as plt
from cat_io import read_ovpf_cat
from seis_proc import stations, edir
from scipy.stats import kurtosis
import obspy.signal
from scipy.fftpack import fft
from obspy.signal.freqattributes import cfrequency_unwindowed
from do_smooth import smooth
from scipy.stats import norm
import matplotlib.mlab as mlab
from do_class_test import stat_mean
from seis_class import plot_confusion_matrix
from do_class_test import r_file

# number of random events to plot
N_EVENTS = 1

# read cat
cat = read_ovpf_cat('MC3_dump.csv')
cat_vf = cat[:, :][cat[:, -1] == 'VF']

# get the types of the events
Ynames = cat_vf[:, 2]
durations = cat_vf[:, 1]
Ynames_unique = np.unique(Ynames)
ntypes = 2
nsta = 1

# make the empty structures to keep the seismograms
data_list = []
data_list.append(np.empty((N_EVENTS, ntypes), dtype=object))

# keep the number of files read and the maximum duration
atype_n = np.zeros(ntypes)
atype_dur = np.zeros(ntypes)

# Read in the data. Be careful that the seismograms may not exist for all the
# events in the catalogue !!

atype = []

###### to plot random seismogram

"""
# for each type :
evtype = np.array([0, 6])

atype = Ynames_unique[0]
print 'Extracting %s'%atype
stimes = cat_vf[:, 0][Ynames == atype]
dur = cat_vf[:, 1][Ynames == atype]
rand_idx = np.random.permutation(len(stimes))
# for each index (random order)
for ista in xrange(1):
    n_ok = 0
    for idx in rand_idx:
        print idx
        # make filename
        sta = stations[ista]
        fname = join(edir, "%s_*.%s.*MSEED"%(stimes[idx].isoformat(),
                     sta.split('.')[0]))
        try:
            st = read(fname)
            data_list[ista][n_ok, 0] = st
            print 'Read %s.'%fname
            if dur[idx] > atype_dur[0]:
                atype_dur[0] = dur[idx]
            n_ok += 1
            if n_ok == N_EVENTS:
                break
        except:
            #print 'File %s does not exist, moving to next file.'%fname
            pass
# have found all files for this type
# save their number
atype_n[0] = n_ok

atype = Ynames_unique[6]
print 'Extracting %s'%atype
stimes = cat_vf[:, 0][Ynames == atype]
dur = cat_vf[:, 1][Ynames == atype]
rand_idx = np.random.permutation(len(stimes))

# for each index (random order)
for ista in xrange(1):
    n_ok = 0
    for idx in rand_idx:
        print idx
        # make filename
        sta = stations[ista]
        fname = join(edir, "%s_*.%s.*MSEED"%(stimes[idx].isoformat(),
                     sta.split('.')[0]))
        try:
            st = read(fname)
            data_list[ista][n_ok, 1] = st
            print 'Read %s.'%fname
            if dur[idx] > atype_dur[1]:
                atype_dur[1] = dur[idx]
            n_ok += 1
            if n_ok == N_EVENTS:
                break
        except:
            #print 'File %s does not exist, moving to next file.'%fname
            pass
# have found all files for this type
# save their number
atype_n[1] = n_ok


"""

######to plot choosen seismogram , put the number in "idx" 



# for the sommital

atype = Ynames_unique[0]
print 'Extracting %s'%atype
stimes = cat_vf[:, 0][Ynames == atype]
dur = cat_vf[:, 1][Ynames == atype]

# for each index (random order)
n_ok = 0
idx = 156 # 192
# make filename
sta = stations[0]
fname = join(edir, "%s_*.%s.*MSEED"%(stimes[idx].isoformat(),
             sta.split('.')[0]))

             
st = read(fname)
data_list[0][n_ok, 0] = st
print 'Read %s.'%fname
if dur[idx] > atype_dur[0]:
    atype_dur[0] = dur[idx]
n_ok += 1


###for the rock fall

atype = Ynames_unique[6]
    
print 'Extracting %s'%atype

stimes = cat_vf[:, 0][Ynames == atype]
dur = cat_vf[:, 1][Ynames == atype]


n_ok = 0
idx = 88 # 461
# make filename
sta = stations[0]
fname = join(edir, "%s_*.%s.*MSEED"%(stimes[idx].isoformat(),
             sta.split('.')[0]))

st = read(fname)
data_list[0][n_ok, 1] = st
print 'Read %s.'%fname
if dur[idx] > atype_dur[1]:
    atype_dur[1] = dur[idx]
n_ok += 1


print atype
print data_list


for itype in xrange(ntypes):    

    # calculate the attribut for each trace
    
    st = data_list[0][0,itype]
    st_dict = st[0]
            
    att = []
    att_names = []
        
    st_filt = st.copy()
    st_filt.filter('bandpass', freqmin=10, freqmax=30, zerophase=True)
        
    data_env = obspy.signal.filter.envelope(st_filt[0].data)
    
    print st_filt[0].stats.npts
    print st[0].stats.npts 
    print len(data_env)
    print st_dict.stats.sampling_rate
    
    dt = st_filt[0].stats.delta 
    npts = st_filt[0].stats.npts
    e_time = (npts-1) * dt
    times = np.linspace(0.0, e_time, num=npts)
                                    
    # Atributs about the frequential part of the signal
    
    nfft = st_filt[0].stats.npts
    fs = st_filt[0].stats.sampling_rate
    
    freq = np.linspace(0, fs, nfft + 1)
    
    signal_fft = fft(st_filt[0].data,nfft) 
    s_fft_nofilt = fft(st_dict.data,nfft)
    
    
    # dominant frequency 
    d_f = freq[np.argmax(abs(signal_fft))]
    att.append(d_f)
    att_names.append("dom_f")
        
    # central frequency
    c_f = cfrequency_unwindowed(st_dict.data, fs=100.0)
    att.append(c_f)
    att_names.append("cent_f")
    
    
    # Energy 10-30hz
    
    E = np.trapz(abs(signal_fft.real)**2)
    att.append(E)
    att_names.append("E")
            
    # AsDec
    data_env_smooth = smooth(data_env,window_len=9,window='hanning') 
    
    a = np.int(np.floor(0.05*len(times))) # 5 percent of the signal's length
    b = data_env_smooth[0:a+1]
    thr = np.mean(b) # treshold
        
    nev = len(times)
    x=[]
    for i in xrange(nev):
        if data_env_smooth[i] >= thr:  # select values above the threshold
            x.append(times[i])
    
    ti = x[0]
    tf = x[-1]
            
            
    tmax = times[np.argmax(data_env_smooth)]
    AsDec = (tmax - ti) / (tf - tmax)
    att.append(AsDec)
    att_names.append("AsDec")
    
    # attributs on the temporal domain
        
    # duration
            
    dur = tf-ti
    att.append(dur)
    att_names.append("DUR")
    print tf, ti

    # Amplitude
    A = np.max(np.abs(st_dict.data)) # no filter
    att.append(A)
    att_names.append("A")
        
    # duration / Amplitude
    dur_A = dur / A
    att.append(dur_A)
    att_names.append ("dur/A")
        
    #kurtosis
    K = kurtosis(st_dict) # no filter
    att.append(K)
    att_names.append ("K")
        
        
    # rapport  max Amplitude / mean of envelope : 
        
        # envelope's mean: 
    mean_data_env = data_env.mean()
    
        # max amplitude : 
    max_data_env = data_env.max()
    
    max_mean = max_data_env / mean_data_env # A = max_A here
    att.append(max_mean)
    att_names.append("maxA_mean")
    
    
    
    ###### do plot
    
    fig = plt.figure(figsize = (10,10))
    fig.subplots_adjust(left = 0.1, bottom = 0.07,
                       right = 0.9, top = 0.96, wspace = 1, hspace = 0.8)

    
    plt.subplot(3,1,1)
    
    print 'Plotting %s'%atype[itype]
    plt.plot(times, st_filt[0].data, 'r', label=st_filt[0].stats.station)
    plt.plot(times, data_env, 'b')
    plt.title('seismogram')
    plt.legend(['filtered signal', 'envelope'])
    
    plt.xlim(0, atype_dur[itype])
    plt.xlabel('Time (s)')
    plt.ylabel('Velocity (um/s)')
             
        
        # kurtosis

            # best fit of data
            
    m = stat_mean(st_dict.data)
    datos = []
    
    for i in xrange(st_dict.stats.npts):
        a = st_filt[0].data[i] - m
        datos.append(a)
    
    plt.subplot(3,1,2)  
      
    (mu, sigma) = norm.fit(datos)

            # the histogram of the data
    n, bins, patches = plt.hist(datos, 60, normed=1, facecolor='green', alpha=0.7)

            # add a 'best fit' line
    
    y = mlab.normpdf( bins, mu, sigma)
    l = plt.plot(bins, y, 'r--', linewidth=2)

            #plot

    plt.xlabel('Smarts')
    plt.ylabel('Probability')
    plt.title('Distribution of Amplitude and best fit gaussian')
    plt.legend(['best fit gaussian','Distribution'])
    plt.grid(True)
    

    filename = 'fig_att%d.png'%itype
    fig.savefig(filename)
    plt.clf()

    #spectrogram
    filename = 'spec%d'%itype
    spec = st_dict.spectrogram(outfile=filename, log = True, title = 'spectrogram')
    
    
# confusion matrix plot
# for the choosen matrix

val, key = r_file('020dict0.dat')

cm = val[0][2]

labels = ['EFF','SOM']
title = 'Linear SVM'
filename = 'linear.png'

plot_confusion_matrix(cm, labels, title, filename)    
