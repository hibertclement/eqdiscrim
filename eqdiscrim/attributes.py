import numpy as np
import signal_proc as sp
from obspy.signal.filter import envelope

def get_AsDec(tr):
    
    # smooth data using a filter of the same length as the sampling rate
    # to give 1-second smoothing window
        
    env = envelope(tr.data)
    smooth_env = sp.smooth(env)
    
    imax = np.argmax(smooth_env)
    AsDec = (imax+1) / float(len(tr.data) - (imax+1))
    
    return AsDec
