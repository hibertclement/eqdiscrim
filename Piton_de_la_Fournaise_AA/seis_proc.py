# -*- coding: utf-8 -*-
import numpy as np
from os import mkdir
from os.path import join, isdir, sep
from obspy.core import read 
import obspy.signal
from obspy.signal.freqattributes import domperiod, cfrequency_unwindowed
from obspy.xseed import Parser
from obspy.core.utcdatetime import UTCDateTime
from glob import glob
from sklearn.preprocessing import LabelEncoder
from scipy.stats import kurtosis
from scipy.fftpack import fft, fftfreq
from do_smooth import smooth

"""
stations = ["BOR.EHZ", "RVL.HHZ"]
datadir = 'data'
edir = join(datadir, 'extracted')
dataless_dir = join(datadir, 'dataless')

# paz = BOR first and RVL second
paz = [{'zeros': [0j, 0j],
        'poles': [-4.3345-4.5487j, -4.3345+4.5487j],
        'gain': 6.999222e+07,
        'sensitivity': 1.0e-6},
       {'zeros': [0j, 0j],
        'poles': [-0.074+0.0740j, -0.074-0.074j,
                  -502.6550+0.0j, -1005.3100+0.0j, -1130.9700+0.0j],
        'gain': 4.679089e+17,
        'sensitivity': 1.0e-6}]
"""

stations = ["RVL.HHZ", "FJS.HHZ"]
datadir = 'data'
edir = join(datadir, 'extracted')
dataless_dir = join(datadir, 'dataless')

paz = [{'zeros': [0j, 0j],
        'poles': [-0.074+0.0740j, -0.074-0.074j,
                  -502.6550+0.0j, -1005.3100+0.0j, -1130.9700+0.0j],
        'gain': 4.679089e+17,
        'sensitivity': 1.0e-6},
       {'zeros': [0j, 0j],
        'poles': [-0.074+0.0740j, -0.074-0.074j,
                  -502.6550+0.0j, -1005.3100+0.0j, -1130.9700+0.0j],
        'gain': 4.679089e+17,
        'sensitivity': 1.0e-6}]

"""
stations = ["GPS.HHZ", "SNE.HHZ"]
datadir = 'data'
edir = join(datadir, 'extracted')
dataless_dir = join(datadir, 'dataless')

paz = [{'zeros': [0j, 0j],
        'poles': [-0.074+0.0740j, -0.074-0.074j,
                  -502.6550+0.0j, -1005.3100+0.0j, -1130.9700+0.0j],
        'gain': 4.679089e+17,
        'sensitivity': 1.0e-6},
       {'zeros': [0j, 0j],
        'poles': [-0.074+0.0740j, -0.074-0.074j,
                  -502.6550+0.0j, -1005.3100+0.0j, -1130.9700+0.0j],
        'gain': 4.679089e+17,
        'sensitivity': 1.0e-6}]
"""
"""
stations = ["BOR.EHZ", "DSO.EHZ"]
datadir = 'data'
edir = join(datadir, 'extracted')
dataless_dir = join(datadir, 'dataless')

paz = [{'zeros': [0j, 0j],
        'poles': [-4.3345-4.5487j, -4.3345+4.5487j],
        'gain': 6.999222e+07,
        'sensitivity': 1.0e-6},
       {'zeros': [0j, 0j],
        'poles': [-4.3345-4.5487j, -4.3345+4.5487j],
        'gain': 6.999222e+07,
        'sensitivity': 1.0e-6}]
"""
"""  
stations = ["ENO.EHZ", "PHR.EHZ"]
datadir = 'data'
edir = join(datadir, 'extracted')
dataless_dir = join(datadir, 'dataless')

paz = [{'zeros': [0j, 0j],
        'poles': [-4.3345-4.5487j, -4.3345+4.5487j],
        'gain': 6.999222e+07,
        'sensitivity': 1.0e-6},
       {'zeros': [0j, 0j],
        'poles': [-4.3345-4.5487j, -4.3345+4.5487j],
        'gain': 6.999222e+07,
        'sensitivity': 1.0e-6}]
        
"""       
pre_filt = (0.005, 0.006, 30.0, 35.0)


def extract_events(start_times, durations, remove_response=False):
    """
    Extracts events from data, using start_times and durations.
    """
    nev = len(start_times)

    if not isdir(edir):
        mkdir(edir)

    for i in xrange(nev):
        stime = start_times[i]
        dur = durations[i]
        for ista in xrange(len(stations)):
            sta = stations[ista]
            st = read(join(datadir, sta, 'PF.*'), starttime=stime,
                      endtime=stime+dur)

            try:
                filename = '%s_%s.MSEED' % (stime.isoformat(), st[0].id)
                print filename
                st.detrend(type='linear')
                if remove_response:
                    st.simulate(paz_remove=paz[ista], pre_filt=pre_filt)
                st.write(join(edir, filename), format="MSEED")
                st_ret = st
            except IndexError:
                pass
            
    return st_ret[0]

def get_events_intersection():
    """
    Three list : list of events for two stations, one list for where BOR is 
    missing, another where RVL is missing.
    """
    otime_dict = {}
        
    for sta in stations:
        
        # keep the station name
        search_exp = '*.%s.*' % sta.split('.')[0] 
        
        # get the filenames
        filenames = glob(join(edir, search_exp))
        
        # get the origin times
        otime_list = []
        for fname in filenames:
            otime_list.append(fname.split('_')[0])
        
        # save them in the station dictionary
        otime_dict[sta] = otime_list   
    
    # get intersection
    keys = otime_dict.keys()
    if len(keys) == 2:
        intersect = set(otime_dict[keys[0]]).intersection(otime_dict[keys[1]])
    else:
        raise NotImplemented('Not yet implemented for more than two stations')

    return intersect

def extract_all_attributs(part_filename_list, nmax=None):
    """
    extract attributes for all events
    """
    nev = len(part_filename_list)
    if nmax is not None:
        nev = np.min([nev, nmax])
    att, att_names = extract_attributs(part_filename_list[0])
    natt = len(att)
   
    # create empty matrix
    X = np.empty((nev, natt), dtype=float)
    
    # fill it one event at a time
    for iev in xrange(nev):
        X[iev, :], att_names = extract_attributs(part_filename_list[iev]) 
    
    return X, att_names   
    

def extract_attributs(part_filename):
    """
    extract all attribut for one event
    """
    st_dict = {}
        
    # read the seismograms for this event and save it to the dictionary
    for sta in stations:
        
        # keep the station name
        search_exp = '%s_*.%s.*' % (part_filename, sta.split('.')[0])
        filename = glob(search_exp)[0]
        st = read(filename)
        st_dict[sta] = st[0]
        
    att = []
    att_names = []
    for sta in stations:
        
        st_filt = st.copy() #st
        st_filt.filter('bandpass', freqmin=10, freqmax=30, zerophase=True)
        
        data_env = obspy.signal.filter.envelope(st_filt[0].data)
        data_env_smooth = smooth(data_env, window_len = 21, window='hanning')

                                        
        # Atributs about the frequential part of the signal
        
        nfft = st_filt[0].stats.npts
        fs = st_filt[0].stats.sampling_rate
        freq = np.linspace(0, fs, nfft + 1)
        
        signal_fft = fft(st_filt[0].data,nfft) 
        s_fft_nofilt = fft(st_dict[sta].data,nfft)
        
        dt = st_filt[0].stats.delta 
        npts = st_filt[0].stats.npts
        e_time = (npts-1) * dt
        times = np.linspace(0.0, e_time, num=npts)
                       
        #spectrogram
        #spec = st_dict[sta].spectrogram()
        
        
        # dominant frequency 
        d_f = freq[np.argmax(abs(signal_fft))] # no filter
        att.append(d_f)
        att_names.append("dom_f")
        
        # central frequency
        c_f = cfrequency_unwindowed(st_dict[sta].data, fs=100.0) # no filter
        att.append(c_f)
        att_names.append("cent_f")
        
        
        # Energy 10-30hz
        
        E = np.trapz(abs(signal_fft.real)**2) # filter
        att.append(E)
        att_names.append("E")
             
        #AsDec
        
        sm_npts = len(data_env_smooth)
        #sm_time = (sm_npts-1) * dt
        #time = np.linspace(0.0, sm_time, num=sm_npts)
        time = np.arange(sm_npts)*dt 
        ti = time[0]
        tf = time[-1]
        """
        n = 0
        if len(data_env) == 0:
            n = n+1
            AsDec = 'NAN'
            att.append(AsDec)
            att_names.append("AsDec")
            
            print n
            
        else :

            emax = np.argmax(data_env_smooth[ti:tf])+ti
            tmax = time[emax]
            #p1 = (tf - tmax)/(tf-ti)  # decay
            #p2 = (tmax - ti)/ (tf-ti)  # growth
        
            AsDec = (tmax - ti) / (tf - tmax)
            att.append(AsDec)
            att_names.append("AsDec")
        """
        emax = np.argmax(data_env_smooth[ti:tf])+ti
        tmax = time[emax]
        #p1 = (tf - tmax)/(tf-ti)  # decay
        #p2 = (tmax - ti)/ (tf-ti)  # growth
        
        AsDec = (tmax - ti) / (tf - tmax)
        att.append(AsDec)
        att_names.append("AsDec")
        
        #duration
        
        dur = st_filt[0].stats.endtime - st_filt[0].stats.starttime
        att.append(dur)
        att_names.append("DUR")
        
        
        """  
    
        # AsDec
        data_env_smooth = smooth(data_env,window_len=9,window='hanning')  # filter
        
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
        
        # duration # filter
                
        dur = tf-ti
        att.append(dur)
        att_names.append("DUR")
        
        """
          
        # Amplitude
        A = np.max(np.abs(st_dict[sta].data)) # no filter
        att.append(A)
        att_names.append("A")
        
        # duration / Amplitude
        dur_A = dur / A
        att.append(dur_A)
        att_names.append ("dur/A")
        

        #kurtosis
        K = kurtosis(st_dict[sta]) # no filter
        att.append(K)
        att_names.append ("K")
        
    
        # rapport  max Amplitude / mean of envelope : 
            
            # envelope's mean: 
        mean_data_env = data_env.mean() #filter
        
            # max amplitude : 
        max_data_env = data_env.max() # filter
        
        max_mean = max_data_env / mean_data_env # A = max_A here
        att.append(max_mean)
        att_names.append("maxA_mean")
        

        
        
    return att, att_names
    
    
def calculate_attributs(st):
    """
   calculate all the attributs for a trace
    """
        
    st_dict = st[0]
        
    att = []
    att_names = []
        
    st_filt = st_dict.copy()
    st_filt.filter('bandpass', freqmin=10, freqmax=30, zerophase=True)
    
    data_env = obspy.signal.filter.envelope(st_filt.data)
    
                                    
    # Atributs about the frequential part of the signal
    
    nfft = len(st_filt)
    fs = 100.0
    freq = np.linspace(0, fs, nfft + 1)
    
    signal_fft = fft(st_filt,nfft) 
    s_fft_nofilt = fft(st_dict,nfft)
    
    dt = st_filt.stats.delta 
    npts = st_filt.stats.npts
    e_time = (npts-1) * dt
    times = np.linspace(0.0, e_time, num=npts)
                   
    #spectrogram
    #spec = st_dict[sta].spectrogram()
    
    
    # dominant frequency 
    d_f = freq[np.argmax(abs(st_dict.data))]
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
        j=0
        if data_env_smooth[i] >= thr:  # select values above the threshold
            x.append(times[i])
            j=j+1    
    
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
            
    # Amplitude
    A = np.max(np.abs(st_dict.data))
    att.append(A)
    att_names.append("A")
    
    # duration / Amplitude
    dur_A = dur / A
    att.append(dur_A)
    att_names.append ("dur/A")
    
    #kurtosis
    K = kurtosis(st_dict)
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
    
    return att, att_names, ti, tf
    
def extract_types(cat, common_fileglob, nmax=None):
    """
    Extract from catalog the types of the common events
    """   
    nev, natt = cat.shape
    nfiles = len(common_fileglob)
    if nmax is not None:
        nfiles = np.min([nfiles, nmax])
    
    Ynames = np.empty(nfiles, dtype=object)
    cat_times = cat[:, 0]
    # for each file we have on disk
    print "Extracting types for %d events..."%nfiles
    for i in xrange(nfiles):
        # extract the start-time as a UTCDateTime object
        st_time = common_fileglob[i].split(sep)[-1]
        st_time = UTCDateTime(st_time)

        # extract from the catalog the event that has this start-time
        diff_times = np.abs(cat_times - st_time)
        st_time_index = np.argmin(diff_times)
        Ynames[i] = cat[st_time_index, 2]
    # now turn the categorical names into integers
    enc = LabelEncoder()
    lab_enc = enc.fit(Ynames)
    int_classes = lab_enc.transform(lab_enc.classes_)
    Y = lab_enc.transform(Ynames)
    
    # keep the dictionnary of class names
    Y_dict = {}
    for i in xrange(len(lab_enc.classes_)):
        Y_dict[int_classes[i]] = lab_enc.classes_[i]
    return Y, Y_dict
  
