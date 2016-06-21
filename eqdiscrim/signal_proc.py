import numpy as np
from obspy.signal.trigger import trigger_onset

def smooth(x, window_len=11, window='hanning'):
    """
    smooth the data using a window with requested size.
        
    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal 
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.
       
    input:
        x: the input signal 
        window_len: the dimension of the smoothing window; should be an odd integer
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
          flat window will produce a moving average smoothing.
 
    output:
       the smoothed signal
         
    """ 
        
    if x.ndim != 1:
        raise ValueError, "smooth only accepts 1 dimension arrays."
   
    if x.size < window_len:
        raise ValueError, "Input vector needs to be bigger than window size."
   
    if window_len<3:
        return x
       
    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError, "Window is one of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'"
   
    s = np.r_[x[window_len-1:0:-1],x,x[-1:-window_len:-1]]
    if window == 'flat': #moving average
        w = np.ones(window_len,'d')
    else:
        w=eval('np.' + window + '(window_len)')
    y = np.convolve(w / w.sum(), s, mode='valid')

    return y[window_len/2+1:-(window_len/2-1)]  


def nextpow2(i):
    n = 1
    while n < i:
        n *= 2
    return n

def find_peaks(sig, thresh):
    itriggers = trigger_onset(sig, thres1=thresh, thres2=thresh)
    n_peaks, n_bid = itriggers.shape
    return n_peaks, itriggers
