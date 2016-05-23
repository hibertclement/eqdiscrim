import numpy as np
import signal_proc as sp
from scipy.stats import kurtosis, skew
from scipy.signal import hilbert, find_peaks_cwt
from obspy.signal.filter import envelope
from obspy.signal.trigger import trigger_onset

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

    #TODO : try using obspy trigger to count peaks above min_peak_height
    CorPeakNumber, itriggers = sp.find_peaks(cor_smooth, min_peak_height)

    # integrate over bands
    ilag_0 = np.argmax(cor_smooth)+1
    ilag_third = ilag_0 + npts/6

    # note that these integrals are flase really (dt is not correct)
    int1 = np.trapz(cor_smooth[ilag_0:ilag_third+1]/max_cor)
    int2 = np.trapz(cor_smooth[ilag_third:]/max_cor)
    int_ratio = int1 / int2

    return CorPeakNumber, int_ratio

def get_freq_band_stuff(tr, FFI=[0.1, 1, 3, 10, 20], FFE=None, corners=2):

    # FFI gives the left (low-frequency) side of the butterworth filter to use
    # FFE gives the right (high-frequency) side of the butterworth filter to use

    sps = tr.stats.sampling_rate
    dt = tr.stats.delta
    NyF = sps / 2.

    nf = len(FFI)
    if FFE is None:
        FFE = np.empty(nf, dtype=float)
        FFE[0 : -1] = FFI[1 : nf]
        FFE[-1] = 0.99 * NyF

    ES = np.empty(nf, dtype=float)
    KurtoF = np.empty(nf, dtype=float)

    for j in xrange(nf):
        tr_filt = tr.copy()
        tr_filt.filter('bandpass', freqmin=FFI[j], freqmax=FFE[j],
                       corners=corners)
        ES[j] = 2 * np.log10(np.trapz(envelope(tr_filt.data), dx=dt))
        KurtoF[j] = kurtosis(tr_filt.data, fisher=False)

    return ES, KurtoF

def get_full_spectrum_stuff(tr):

    sps = tr.stats.sampling_rate
    NyF = sps / 2.0

    data = tr.data
    npts = tr.stats.npts
    n = sp.nextpow2(2 * npts - 1)
    df = n / 2 * NyF
    Freq1 = np.linspace(0, NyF, df)

    FFTdata = 2 * np.abs(np.fft.fft(data, n=n)) / float(npts * npts)
    FFTsmooth = sp.smooth(FFTdata[0:len(FFTdata)/2])
    FFTsmooth_norm = FFTsmooth / max(FFTsmooth)

    MeanFFT = np.mean(FFTsmooth_norm)
    MedianFFT = np.median(FFTsmooth_norm)
    VarFFT = np.var(FFTsmooth_norm, ddof=1)
    MaxFFT = np.max(FFTsmooth)
    iMaxFFT = np.argmax(FFTsmooth)
    FmaxFFT = Freq1[iMaxFFT]

    xCenterFFT = np.sum((np.arange(len(FFTsmooth_norm))) *
                        FFTsmooth_norm) / np.sum(FFTsmooth_norm)
    i_xCenterFFT = int(np.round(xCenterFFT))

    xCenterFFT_1quart = np.sum((np.arange(i_xCenterFFT+1)) *
                               FFTsmooth_norm[0:i_xCenterFFT+1]) /\
        np.sum(FFTsmooth_norm[0:i_xCenterFFT+1])
    i_xCenterFFT_1quart = int(np.round(xCenterFFT_1quart))

    xCenterFFT_3quart = np.sum((np.arange(len(FFTsmooth_norm) -
                                         i_xCenterFFT)) *
                              FFTsmooth_norm[i_xCenterFFT:]) /\
        np.sum(FFTsmooth_norm[i_xCenterFFT:]) + i_xCenterFFT+1
    i_xCenterFFT_3quart = int(np.round(xCenterFFT_3quart))

    FCentroid = Freq1[i_xCenterFFT]
    Fquart1 = Freq1[i_xCenterFFT_1quart]
    Fquart3 = Freq1[i_xCenterFFT_3quart]

    min_peak_height = 0.75
    NpeakFFT, ipeaks = sp.find_peaks(FFTsmooth_norm, min_peak_height)

    np, nb = ipeaks.shape
    sum_peaks = 0.
    for i in xrange(np):
        sum_peaks += max(FFTsmooth_norm[ipeaks[i,0] : ipeaks[i, 1]])
    MeanPeaksFFT = sum_peaks / float(n_peaks)

    npts = len(FFTsmooth_norm)
    # beware : integrals are bogus
    E1FFT = np.trapz(FFTsmooth_norm[0:npts/4], dx=df)
    E2FFT = np.trapz(FFTsmooth_norm[npts/4-1:2*npts/4], dx=df)
    E3FFT = np.trapz(FFTsmooth_norm[2*npts/4-1:3*npts/4], dx=df)
    E4FFT = np.trapz(FFTsmooth_norm[3*npts/4-1:npts], dx=df)

    moment = np.empty(3, dtype=float)

    for j in xrange(3):
        moment[j] = np.sum(Freq1**j * FFTsmooth_norm[0:n/2]**2)
    gamma1 = moment[1]/moment[0]
    gamma2 = np.sqrt(moment[2]/moment[0])
    gammas = np.sqrt(np.abs(gamma1[i]**2 - gamma2[i]**2))

    return MeanFFT, MaxFFT, FmaxFFT, MedianFFT, VarFFT, FCentroid, Fquart1,\
        Fquart3, NpeakFFT, MeanPeaksFFT, E1FFT, E2FFT, E3FFT, E4FFT, gamma1,\
        gamma2, gammas


