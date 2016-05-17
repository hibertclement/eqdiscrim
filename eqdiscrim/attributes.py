import numpy as np
from scipy.signal import lfilter, hilbert


def envelope(tr):

    env = np.abs(hilbert(tr.data))

    return env

def get_AsDec(tr, env):
    
    # smooth data using a filter of the same length as the sampling rate
    # to give 1-second smoothing window
        
    sps = tr.stats.sampling_rate
    strong_filter = np.ones(int(sps)) / float(sps)
    smooth_env = lfilter(strong_filter, 1, env)
    
    imax = np.argmax(smooth_env)
    AsDec = (imax+1) / float(len(tr.data) - (imax+1))
    
    return AsDec