import numpy as np
from obspy.core import Trace
from scipy.signal import sweep_poly


def gen_gaussian_noise(dt, npts, amp):
    """
    Generates synthetic noise with gaussian amplitude distribution around zero.
    """

    # create noise vector
    n_vect = np.random.randn(npts) * amp

    # create corresponding obspy trace
    header = {'delta': dt, 'npts': npts, 'station': 'SYN', 'network': 'XX',
              'channel': 'HHZ', 'location': '00'}
    tr = Trace(data=n_vect, header=header)
 
    # return
    return tr
    
def modulate_trace_triangle(tr, imax_fraction, max_amp):
    """
    Multiplies the trace by a triangle function, whose maximum max_amp occurs
    at a time after the start of the trace corresponding to imax_fraction of the
    trace length.
    """
    #import pdb; pdb.set_trace()
    
    # get trace attributes
    duration = tr.stats.endtime - tr.stats.starttime
    dt = tr.stats.delta
    npts = tr.stats.npts
    
    # create the modulation function
    t_max = duration * imax_fraction
    it_max = int(t_max / dt)
    slope_up = max_amp / t_max
    slope_down = max_amp / (duration - t_max)

    t_vect = np.arange(npts) * dt
    modulation = np.empty(npts, dtype=float)
    for i in xrange(npts):
        t = i * dt
        if t <= t_max:
            modulation[i] = t * slope_up
        else:
            modulation[i] = max_amp - (t-t_max) * slope_down
    
    # apply modulation
    max_data = max(abs(tr.data))
    tr.data *= modulation / max_data
    
    # return modulated data
    return tr, modulation
    
def gen_sweep_poly(dt, npts, poly_coeffs = [0.05, -0.75, 2.5, 5.0]):

    p = np.poly1d(poly_coeffs)
    t = np.arange(npts) * dt
    w = sweep_poly(t, p)

    # create corresponding obspy trace
    header = {'delta': dt, 'npts': npts, 'station': 'SYN', 'network': 'XX',
              'channel': 'HHZ', 'location': '00'}
    tr = Trace(data=w, header=header)
 
    # return
    return tr

def gen_repeat_signal(dt, npts, npts_sig, amp_sig, n_sig=3):

    full_sig = np.zeros(npts)
    lag = npts / (2*n_sig) - npts_sig / 2
    step = npts / n_sig

    t_sig = np.arange(npts_sig) * dt
    T = (npts_sig-1) * dt
    sig = np.sin(2 * np.pi / T * t_sig) * amp_sig

    for i in xrange(n_sig):
        i_start = i * step + lag
        full_sig[i_start : i_start + npts_sig] = sig[:]
        
    # create corresponding obspy trace
    header = {'delta': dt, 'npts': npts, 'station': 'SYN', 'network': 'XX',
              'channel': 'HHZ', 'location': '00'}
    tr = Trace(data=full_sig, header=header)

    return tr

