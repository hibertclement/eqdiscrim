import numpy as np
import signal_proc as sp
from scipy.stats import kurtosis, skew
from scipy.signal import hilbert, find_peaks_cwt
from obspy.signal.filter import envelope

def get_AsDec(tr, env=None):
    
    # smooth data using a filter of the same length as the sampling rate
    # to give 1-second smoothing window
        
    if env is None:
        env = envelope(tr.data)

    smooth_env = sp.smooth(env)
    
    imax = np.argmax(smooth_env)
    AsDec = (imax+1) / float(len(tr.data) - (imax+1))
    
    return AsDec

def get_Duration(tr):

    # get straight file duration
    # TODO : make this more intelligent using picking etc.

    return tr.stats.endtime - tr.stats.starttime

def get_RappStuff(tr, env=None):

    # get max over mean / median / std
    if env is None:
        env = envelope(tr.data)

    max_env = max(env)
    smooth_norm_env = sp.smooth(env/max_env)

    RappMaxMean = 1. / np.mean(smooth_norm_env)
    RappMaxMedian = 1. / np.median(smooth_norm_env)
    RappMaxStd = 1. / np.std(smooth_norm_env)
    
    return RappMaxMean, RappMaxMedian, RappMaxStd

def get_KurtoSkew(tr, env=None):

    if env is None:
        env = envelope(tr.data)

    max_env = max(env)
    smooth_norm_env = sp.smooth(env / max_env)

    max_sig = max(tr.data)
    norm_sig = tr.data / max_sig

    KurtoEnv = kurtosis(smooth_norm_env, fisher=False)
    KurtoSig = kurtosis(norm_sig, fisher=False)

    SkewnessEnv = skew(smooth_norm_env)
    SkewnessSig = skew(norm_sig) 

    return KurtoEnv, KurtoSig, SkewnessEnv, SkewnessSig

    
def get_CorrStuff(tr):

    min_peak_height = 0.4
    # This value is sligtly greater than the matlab one, to account for
    # differences in floating precision

    cor = np.correlate(tr.data, tr.data, mode='full')
    cor = cor / np.max(cor)

    # find number of peaks
    cor_env = np.abs(hilbert(cor))
    cor_smooth = sp.smooth(cor_env)
    npts = len(cor_smooth)
    max_cor = np.max(cor_smooth)

    print '2'

    #TODO : try using obspy trigger to count peaks above min_peak_height
    ipeaks = find_peaks_cwt(cor_smooth, np.arange(1, npts/2))
    n_peaks = 0
    for ip in ipeaks:
        if cor_smooth[ip] > min_peak_height:
            n_peaks += 1
    CorPeakNumber = n_peaks

    print '3'
    # integrate over bands
    ilag_0 = np.argmax(cor_smooth)+1
    ilag_third = ilag_0 + npts/6

    print '4'
    # note that these integrals are flase really (dt is not correct)
    int1 = np.trapz(cor_smooth[ilag_0:ilag_third+1]/max_cor)
    int2 = np.trapz(cor_smooth[ilag_third:]/max_cor)
    int_ratio = int1 / int2
    print '5'

    return CorPeakNumber, int_ratio


