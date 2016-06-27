import numpy as np
import signal_proc as sp
from scipy.stats import kurtosis, skew
from scipy.signal import hilbert, find_peaks_cwt
from obspy.signal.filter import envelope

att_names_single_station_1D = list(['AsDec', 'Duration', 'RappMaxMean',
                      'RappMaxMedian', 'RappMaxStd', 'KurtoEnv', 'KurtoSig',
                      'SkewnessEnv', 'SkewnessSig', 'CorPeakNumber',
                      'int_ratio', 'ES1', 'ES2', 'ES3', 'ES4', 'ES5',
                      'KurtoF1', 'KurtoF2', 'KurtoF3', 'KurtoF4', 'KurtoF5',
                      'MeanFFT', 'MaxFFT', 'FmaxFFT', 'MedianFFT', 'VarFFT',
                      'FCentroid', 'Fquart1', 'Fquart3', 'NpeakFFT',
                      'MeanPeaksFFT', 'E1FFT', 'E2FFT', 'E3FFT', 'E4FFT',
                      'gamma1', 'gamma2', 'gammas', 'MaxAmp', 'DurOverAmp'])

def get_all_single_station_attributes(st):

    NaN_value = -12345.0
    min_length = 11

    # set attribute names
    att_names = att_names_single_station_1D

    if st is None or len(st) == 0:
        # return names of attributes and NaN values
        att = np.ones((1, len(att_names)), dtype=float) * np.nan
        return att, att_names

    if st[0].stats.npts < min_length:
        # return names of attributes and NaN values
        att = np.ones((1, len(att_names)), dtype=float) * np.nan
        return att, att_names

    # create the amplitude trace and its envelope
    if len(st) == 3:
        att_names.extend(list(['rectilinP', 'azimuthP', 'dipP', 'Plani']))
        amp_data = np.sqrt(st[0].data * st[0].data + st[1].data * st[1].data +
                            st[2].data * st[2].data)
        amp_trace = st.select(component="Z")[0].copy()
        amp_trace.data = amp_data
    else:
        try:
            amp_trace = st.select(component="Z")[0]
        except IndexError:
            att = np.ones((1, len(att_names)), dtype=float) * np.nan
            return att, att_names
    env = envelope(amp_trace.data)

    if len(st) == 3:
        att_names.extend(list(['rectilinP', 'azimuthP', 'dipP', 'Plani']))
    att = np.empty((1, len(att_names)), dtype=float)

    att[0, 0:1] = get_AsDec(amp_trace, env)
    att[0, 1:2] = get_Duration(amp_trace)
    att[0, 2:5] = get_RappStuff(amp_trace, env)
    att[0, 5:9] = get_KurtoSkew(amp_trace, env)
    att[0, 9:11] = get_CorrStuff(amp_trace)
    ES, KurtoF = get_freq_band_stuff(amp_trace)
    att[0, 11:16] = ES[:]
    att[0, 16:21] = KurtoF[:]
    att[0, 21:38] = get_full_spectrum_stuff(amp_trace)
    att[0, 38:40] = get_AmpStuff(amp_trace)
    if len(st) == 3:
        att[0, 40:44] = get_polarization_stuff(st, env)

    return att, att_names

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
    Freq1 = np.linspace(0, 1, n/2) * NyF

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

    npeaks, nb = ipeaks.shape
    sum_peaks = 0.
    for i in xrange(npeaks):
        i1 = ipeaks[i, 0]
        i2 = ipeaks[i, 1]
        if i2 > i1:
            sum_peaks += max(FFTsmooth_norm[i1 : i2])
    MeanPeaksFFT = sum_peaks / float(npeaks)

    npts = len(FFTsmooth_norm)
    E1FFT = np.trapz(FFTsmooth_norm[0:npts/4], dx=df)
    E2FFT = np.trapz(FFTsmooth_norm[npts/4-1:2*npts/4], dx=df)
    E3FFT = np.trapz(FFTsmooth_norm[2*npts/4-1:3*npts/4], dx=df)
    E4FFT = np.trapz(FFTsmooth_norm[3*npts/4-1:npts], dx=df)

    moment = np.empty(3, dtype=float)

    for j in xrange(3):
        moment[j] = np.sum(Freq1**j * FFTsmooth_norm[0:n/2]**2)
    gamma1 = moment[1]/moment[0]
    gamma2 = np.sqrt(moment[2]/moment[0])
    gammas = np.sqrt(np.abs(gamma1**2 - gamma2**2))

    return MeanFFT, MaxFFT, FmaxFFT, MedianFFT, VarFFT, FCentroid, Fquart1,\
        Fquart3, NpeakFFT, MeanPeaksFFT, E1FFT, E2FFT, E3FFT, E4FFT, gamma1,\
        gamma2, gammas

def get_AmpStuff(tr):

    MaxAmp = max(abs(tr.data)) * 1000.
    dur = tr.stats.endtime - tr.stats.starttime
    DurOverAmp = dur/MaxAmp 

    return MaxAmp, DurOverAmp

def get_polarization_stuff(st, env):

    Ztrace = st.select(component="Z")[0]
    Ntrace = st.select(component="N")[0]
    Etrace = st.select(component="E")[0]
    smooth_env = sp.smooth(env)
    imax = np.argmax(smooth_env)
    end_window = int(np.round(imax/3.))

    # normalise to get decent numbers
    maxZ = max(abs(Ztrace.data))
    xP = Etrace.data[0:end_window] / maxZ
    yP = Ntrace.data[0:end_window] / maxZ
    zP = Ztrace.data[0:end_window] / maxZ

    try:
        MP = np.cov(np.array([xP, yP, zP]))
        w, v = np.linalg.eig(MP)
    except np.linalg.linalg.LinAlgError:
        return np.NaN, np.NaN, np.NaN, np.NaN

    indexes = np.argsort(w)
    DP = w[indexes]
    pP = v[:, indexes]

    rectilinP = 1 - ((DP[0] + DP[1]) / (2*DP[2]))
    azimuthP = np.arctan(pP[1, 2] / pP[0, 2]) * 180./np.pi
    dipP = np.arctan(pP[2, 2] / np.sqrt(pP[1, 2]**2 + pP[0, 2]**2)) * 180/np.pi
    Plani = 1 - (2 * DP[0]) / (DP[1] + DP[2])

    return rectilinP, azimuthP, dipP, Plani


