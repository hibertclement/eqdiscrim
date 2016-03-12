import numpy as np
from scipy.signal import hilbert, lfilter, filtfilt, find_peaks_cwt, butter
from scipy.stats import kurtosis, skew

NATT = 61
CoefSmooth = 3

def calculate_all_attributes(obspy_stream):

    # make sure is in right order (Z then horizontals)
    
    obspy_stream.sort(keys=['channel'], reverse=True)
    all_attributes = np.empty((1, NATT), dtype=float)

    dur = duration(obspy_stream)
    env = envelope(obspy_stream)

    TesMEAN, TesMEDIAN, TesSTD = get_TesStuff(env)
    RappMaxMean, RappMaxMedian = get_RappMaxStuff(TesMEAN, TesMEDIAN)
    AsDec, RmsDecAmpEnv = get_AsDec(obspy_stream, env)
    KurtoEnv, KurtoSig, SkewnessEnv, SkewnessSig = get_KurtoSkewStuff(obspy_stream, env)
    CorPeakNumber, INT1, INT2, INT_RATIO = get_CorrStuff(obspy_stream)
    ES, KurtoF = get_freq_band_stuff(obspy_stream)
    MeanFFT, MaxFFT, FmaxFFT, MedianFFT, VarFFT, FCentroid, Fquart1, Fquart3,\
        NpeakFFT, MeanPeaksFFT, E1FFT, E2FFT, E3FFT, E4FFT, gamma1, gamma2,\
        gammas = get_full_spectrum_stuff(obspy_stream)
    rectilinP, azimuthP, dipP, Plani = get_polarization_stuff(obspy_stream, env)

    # waveform
    all_attributes[0, 0] = np.mean(duration(obspy_stream))
    all_attributes[0, 1] = np.mean(RappMaxMean)
    all_attributes[0, 2] = np.mean(RappMaxMedian)
    all_attributes[0, 3] = np.mean(AsDec)
    all_attributes[0, 4] = np.mean(KurtoSig)
    all_attributes[0, 5] = np.mean(KurtoEnv)
    all_attributes[0, 6] = np.mean(np.abs(SkewnessSig))
    all_attributes[0, 7] = np.mean(np.abs(SkewnessEnv))
    all_attributes[0, 8] = np.mean(CorPeakNumber)
    all_attributes[0, 9] = np.mean(INT1)
    all_attributes[0, 10] = np.mean(INT2)
    all_attributes[0, 11] = np.mean(INT_RATIO)
    all_attributes[0, 12] = np.mean(ES[:, 0])
    all_attributes[0, 13] = np.mean(ES[:, 1])
    all_attributes[0, 14] = np.mean(ES[:, 2])
    all_attributes[0, 15] = np.mean(ES[:, 3])
    all_attributes[0, 16] = np.mean(ES[:, 4])
    all_attributes[0, 17] = np.mean(KurtoF[:, 0])
    all_attributes[0, 18] = np.mean(KurtoF[:, 1])
    all_attributes[0, 19] = np.mean(KurtoF[:, 2])
    all_attributes[0, 20] = np.mean(KurtoF[:, 3])
    all_attributes[0, 21] = np.mean(KurtoF[:, 4])
    all_attributes[0, 22] = np.mean(RmsDecAmpEnv)

    # spectral
    all_attributes[0, 23] = np.mean(MeanFFT)
    all_attributes[0, 24] = np.mean(MaxFFT)
    all_attributes[0, 25] = np.mean(FmaxFFT)
    all_attributes[0, 26] = np.mean(FCentroid)
    all_attributes[0, 27] = np.mean(Fquart1)
    all_attributes[0, 28] = np.mean(Fquart3)
    all_attributes[0, 29] = np.mean(MedianFFT)
    all_attributes[0, 30] = np.mean(VarFFT)
    all_attributes[0, 31] = np.mean(NpeakFFT)
    all_attributes[0, 32] = np.mean(MeanPeaksFFT)
    all_attributes[0, 33] = np.mean(E1FFT)
    all_attributes[0, 34] = np.mean(E2FFT)
    all_attributes[0, 35] = np.mean(E3FFT)
    all_attributes[0, 36] = np.mean(E4FFT)
    all_attributes[0, 37] = np.mean(gamma1)
    all_attributes[0, 38] = np.mean(gamma2)
    all_attributes[0, 39] = np.mean(gammas)

    # polarisation
    all_attributes[0, 57] = rectilinP
    all_attributes[0, 58] = azimuthP
    all_attributes[0, 59] = dipP
    all_attributes[0, 60] = Plani

    return all_attributes

def duration(obspy_stream):

    ntr = len(obspy_stream)
    dur = np.empty(ntr, dtype=float)

    for i in xrange(ntr):
        tr = obspy_stream[i]
        dur[i] = len(tr.data) / tr.stats.sampling_rate
        # la duree ci-dessus est fausse... peut-on changer avec celle ci-dessous ?
        #dur[i] = tr.stats.endtime - tr.stats.starttime
    
    return dur

def envelope(obspy_stream):

    ntr = len(obspy_stream)
    env = np.empty(ntr, dtype=object)

    for i in xrange(ntr):
        tr = obspy_stream[i]
        env[i] = np.abs(hilbert(tr.data))

    return env

def get_TesStuff(env):
    
    ntr = len(env)
    TesMEAN = np.empty(ntr, dtype=float)
    TesMEDIAN = np.empty(ntr, dtype=float)
    TesSTD = np.empty(ntr, dtype=float)

    light_filter = np.ones(CoefSmooth) / float(CoefSmooth)

    for i in xrange(ntr):
        env_max = np.max(env[i])
        tmp = lfilter(light_filter, 1, env[i]/env_max)
        TesMEAN[i] = np.mean(tmp)
        TesMEDIAN[i] = np.median(tmp)
        TesSTD[i] = np.std(tmp)

    return TesMEAN, TesMEDIAN, TesSTD

def get_RappMaxStuff(TesMEAN, TesMEDIAN):

    # ce calcul est hautement improbable
    # je propose 1./TesMEAN et 1./TesMEDIAN

    npts = len(TesMEAN)
    RappMaxMean = np.empty(npts, dtype=float)
    RappMaxMedian = np.empty(npts, dtype=float)

    for i in xrange(npts):
        RappMaxMean[i] = 1./np.mean(TesMEAN[0:i+1])
        RappMaxMedian[i] = 1./np.mean(TesMEDIAN[0:i+1])

    return RappMaxMean, RappMaxMedian

def get_AsDec(st, env):

    sps = st[0].stats.sampling_rate
    strong_filter = np.ones(int(sps)) / float(sps)

    ntr = len(st)
    
    AsDec = np.empty(ntr, dtype=float)
    RmsDecAmpEnv = np.empty(ntr, dtype=float)

    for i in xrange(ntr):
        smooth_env = lfilter(strong_filter, 1, env[i])
        imax = np.argmax(smooth_env)
        AsDec[i] = (imax+1) / float(len(st[i].data) - (imax+1))
        dec = st[i].data[imax:]
        lendec = len(dec)
        # I don't know what this is really, but it does not look like an RMS !!
        RmsDecAmpEnv[i] = np.abs(np.mean(np.abs(hilbert(dec / np.max(st[i].data))) -\
            (1 - ( (1 / float(lendec)) * (np.arange(lendec)+1) )))) 
    
    return AsDec, RmsDecAmpEnv

def get_KurtoSkewStuff(st, env):

    ntr = len(st)

    KurtoEnv = np.empty(ntr, dtype=float)
    KurtoSig = np.empty(ntr, dtype=float)
    SkewnessEnv = np.empty(ntr, dtype=float)
    SkewnessSig = np.empty(ntr, dtype=float)

    light_filter = np.ones(CoefSmooth) / float(CoefSmooth)

    for i in xrange(ntr):
        env_max = np.max(env[i])
        data_max = np.max(st[i].data)
        tmp = lfilter(light_filter, 1, env[i]/env_max)
        KurtoEnv[i] = kurtosis(tmp, fisher=False)
        SkewnessEnv[i] = skew(tmp)
        KurtoSig[i] = kurtosis(st[i].data / data_max, fisher=False)
        SkewnessSig[i] = skew(st[i].data / data_max)

    return KurtoEnv, KurtoSig, SkewnessEnv, SkewnessSig

def get_CorrStuff(st):

    ntr = len(st)

    sps = st[0].stats.sampling_rate
    strong_filter = np.ones(int(sps)) / float(sps)
    min_peak_height = 0.405
    # This value is sligtly greater than the matlab one, to account for
    # differences in floating precision

    CorPeakNumber = np.empty(ntr, dtype=int)
    INT1 = np.empty(ntr, dtype=float)
    INT2 = np.empty(ntr, dtype=float)
    INT_RATIO = np.empty(ntr, dtype=float)

    for i in xrange(ntr):
        cor = np.correlate(st[i].data, st[i].data, mode='full')
        cor = cor / np.max(cor)

        # find number of peaks
        cor_env = np.abs(hilbert(cor))
        cor_smooth = filtfilt(strong_filter, 1, cor_env)
        cor_smooth2 = filtfilt(strong_filter, 1, cor_smooth/np.max(cor_smooth))
        ipeaks = find_peaks_cwt(cor_smooth2, np.arange(1, int(sps)))
        n_peaks = 0
        for ip in ipeaks:
            if cor_smooth2[ip] > min_peak_height:
                n_peaks += 1
        CorPeakNumber[i] = n_peaks

        # integrate over bands
        npts = len(cor_smooth)
        ilag_0 = np.argmax(cor_smooth)+1
        ilag_third = ilag_0 + npts/6

        # note that these integrals are flase really (dt is not correct)
        max_cor = np.max(cor_smooth)
        int1 = np.trapz(cor_smooth[ilag_0 : ilag_third+1]/max_cor)
        int2 = np.trapz(cor_smooth[ilag_third : ]/max_cor)
        int_ratio = int1 / int2

        INT1[i] = int1
        INT2[i] = int2
        INT_RATIO[i] = int_ratio


    return CorPeakNumber, INT1, INT2, INT_RATIO

def get_freq_band_stuff(st):

    sps = st[0].stats.sampling_rate
    NyF = sps / 2.

    FFI = np.array([0.1, 1, 3, 10, 20])
    FFE = np.array([1, 3, 10, 20, 0.99*NyF])

    ntr = len(st)
    nf = len(FFI)

    ES = np.empty((ntr, nf), dtype=float)
    KurtoF = np.empty((ntr, nf), dtype=float)

    for i in xrange(ntr):
        for j in xrange(nf):
            #import pdb; pdb.set_trace()
            #Fb, Fa = butter(2, [FFI[j]/NyF, FFE[j]/NyF], 'bandpass')
            #data_filt = filtfilt(Fb, Fa, tr.data)  # This filter is broken !!
            tr = st[i].copy()
            data_filt = tr.filter('bandpass', freqmin=FFI[j], freqmax=FFE[j], corners=2, zerophase=True)
            # this integral is also bogus...
            ES[i, j] = np.log10(np.trapz(np.abs(hilbert(data_filt))))
            KurtoF[i, j] = kurtosis(data_filt, fisher=False)
            
    return ES, KurtoF

def get_full_spectrum_stuff(st):

    sps = st[0].stats.sampling_rate
    NyF = sps / 2.0
    
    ntr = len(st)
    MeanFFT = np.empty(ntr, dtype=float)
    MaxFFT = np.empty(ntr, dtype=float)
    FmaxFFT = np.empty(ntr, dtype=float)
    MedianFFT = np.empty(ntr, dtype=float)
    VarFFT = np.empty(ntr, dtype=float)
    FCentroid = np.empty(ntr, dtype=float)
    Fquart1 = np.empty(ntr, dtype=float)
    Fquart3 = np.empty(ntr, dtype=float)
    NpeakFFT = np.empty(ntr, dtype=float)
    MeanPeaksFFT = np.empty(ntr, dtype=float)
    E1FFT = np.empty(ntr, dtype=float)
    E2FFT = np.empty(ntr, dtype=float)
    E3FFT = np.empty(ntr, dtype=float)
    E4FFT = np.empty(ntr, dtype=float)
    gamma1 = np.empty(ntr, dtype=float)
    gamma2 = np.empty(ntr, dtype=float)
    gammas = np.empty(ntr, dtype=float)

    b = np.ones(300) / 300.0

    for i in xrange(ntr):
        data = st[i].data
        npts = st[i].stats.npts
        n = nextpow2(2*npts-1)
        dt = st[i].stats.delta
        Freq1 = np.linspace(0, 1, n/2) * NyF

        FFTdata = 2 * np.abs(np.fft.fft(data, n=n)) / float(npts * npts)
        FFTsmooth = lfilter(b, 1, FFTdata[0 : len(FFTdata)/2])
        FFTsmooth_norm = FFTsmooth / max(FFTsmooth)

        MeanFFT[i] = np.mean(FFTsmooth_norm)
        MedianFFT[i] = np.median(FFTsmooth_norm)
        VarFFT[i] = np.var(FFTsmooth_norm, ddof=1)
        MaxFFT[i] = np.max(FFTsmooth)
        iMaxFFT = np.argmax(FFTsmooth)
        FmaxFFT[i] = Freq1[iMaxFFT]

        xCenterFFT = np.sum((np.arange(len(FFTsmooth_norm)) ) * FFTsmooth_norm) /\
            np.sum(FFTsmooth_norm)
        i_xCenterFFT = int(np.round(xCenterFFT))

        xCenterFFT_1quart = np.sum((np.arange(i_xCenterFFT+1)) *\
            FFTsmooth_norm[0 : i_xCenterFFT+1]) /\
            np.sum(FFTsmooth_norm[0 : i_xCenterFFT+1])
        i_xCenterFFT_1quart = int(np.round(xCenterFFT_1quart))

        xCenterFFT_3quart = np.sum((np.arange(len(FFTsmooth_norm) - i_xCenterFFT)) *\
            FFTsmooth_norm[i_xCenterFFT :]) /\
            np.sum(FFTsmooth_norm[i_xCenterFFT :]) + i_xCenterFFT+1
        i_xCenterFFT_3quart = int(np.round(xCenterFFT_3quart))

        FCentroid[i] = Freq1[i_xCenterFFT]
        Fquart1[i] = Freq1[i_xCenterFFT_1quart]
        Fquart3[i] = Freq1[i_xCenterFFT_3quart]

        ipeaks = find_peaks_cwt(FFTsmooth_norm, np.arange(n/8, n/2))
        min_peak_height = 0.75
        n_peaks = 0
        sum_peaks = 0.
        for ip in ipeaks:
            if FFTsmooth_norm[ip] > min_peak_height:
                n_peaks += 1
                sum_peaks += FFTsmooth_norm[ip]

        NpeakFFT[i] = n_peaks
        MeanPeaksFFT[i] = sum_peaks / float(n_peaks)

        npts = len(FFTsmooth_norm)
        E1FFT[i] = np.trapz(FFTsmooth_norm[0 : npts/4])
        E2FFT[i] = np.trapz(FFTsmooth_norm[npts/4-1 : 2*npts/4])
        E3FFT[i] = np.trapz(FFTsmooth_norm[2*npts/4-1 : 3*npts/4])
        E4FFT[i] = np.trapz(FFTsmooth_norm[3*npts/4-1 : npts])
 
        moment = np.empty(3, dtype=float)

        for j in xrange(3):
            moment[j] = np.sum( Freq1**j * FFTsmooth_norm[0:n/2]**2)
        gamma1[i] = moment[1]/moment[0]
        gamma2[i] = np.sqrt(moment[2]/moment[0])
        gammas[i] = np.sqrt(np.abs(gamma1[i]**2 - gamma2[i]**2))
    
    return MeanFFT, MaxFFT, FmaxFFT, MedianFFT, VarFFT, FCentroid, Fquart1, \
           Fquart3, NpeakFFT, MeanPeaksFFT, E1FFT, E2FFT, E3FFT, E4FFT, gamma1,\
           gamma2, gammas

def get_polarization_stuff(st, env):

    sps = st[0].stats.sampling_rate
    strong_filter = np.ones(int(sps)) / float(sps)
    smooth_env = lfilter(strong_filter, 1, env[0])
    imax = np.argmax(smooth_env)
    end_window = int(np.round(imax/3.))

    xP = st[2].data[0:end_window]
    yP = st[1].data[0:end_window]
    zP = st[0].data[0:end_window]

    MP = np.cov(np.array([xP, yP, zP]))
    w, v = np.linalg.eig(MP)

    indexes = np.argsort(w)
    DP = w[indexes]
    pP = v[:, indexes]

    rectilinP = 1 - ( (DP[0] + DP[1]) / (2*DP[2]) )
    azimuthP = np.arctan(pP[1, 2] / pP[0, 2]) * 180./np.pi
    dipP = np.arctan(pP[2, 2] / np.sqrt(pP[1, 2]**2 + pP[0, 2]**2)) * 180/np.pi
    Plani = 1 - (2 * DP[0]) / (DP[1] + DP[2])

    #import pdb; pdb.set_trace()
    return rectilinP, azimuthP, dipP, Plani


def nextpow2(i):
    n = 1
    while n < i: n *= 2
    return n
