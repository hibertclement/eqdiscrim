import numpy as np
from scipy.signal import hilbert, lfilter

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
    AsDec = get_AsDec(obspy_stream, env)

    all_attributes[0, 0] = np.mean(duration(obspy_stream))
    all_attributes[0, 1] = np.mean(RappMaxMean)
    all_attributes[0, 2] = np.mean(RappMaxMedian)
    all_attributes[0, 3] = np.mean(AsDec)

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

    for i in xrange(ntr):
        smooth_env = lfilter(strong_filter, 1, env[i])
        imax = np.argmax(smooth_env)
        AsDec[i] = (imax+1) / float(len(st[i].data) - (imax+1))
    
    return AsDec
