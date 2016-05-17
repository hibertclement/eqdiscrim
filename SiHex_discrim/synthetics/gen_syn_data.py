import numpy as np
from os.path import join
from obspy.core import Trace, UTCDateTime
from scipy.stats import uniform, poisson, norm


def generate_synthetic_gaussian_tremor(T0, dt, npts, amplitude, noise, mu, sigma):
    """
    Generates a synthetic gaussian tremor with single period T0, time-step dt,
    datalength (npts-1)*dt (in seconds) and given amplitude. Shape of envelope
    is gaussian-like.

    Returns an obspy trace with minimal header information.
    """

    # simple check for nyquist
    if T0 < 2*dt:
        raise ValueError('T0 should be at least 2*dt')

    # set up the time vector
    t = np.linspace(0, (npts-1)*dt, npts)
    n_vect = np.random.randn(npts)*noise

    # signal = sin(2*pi/T0)
    signal = np.sin(t * 2*np.pi / T0)

    # modulation = exp(-(t-tmid)^2/sigma^2)
    modulation = amplitude * np.exp(-(t-mu)**2 / sigma**2)

    # create obspy trace
    header = {'delta': dt, 'npts': npts, 'station': 'SYN', 'network': 'XX',
              'channel': 'HHZ', 'location': '00'}
    tr = Trace(data=signal*modulation+n_vect, header=header)

    # return
    return tr

def generate_synthetic_exponential_tremor(T0, dt, npts, amplitude, noise, tstart, tlen):
    """
    Generates a synthetic tremor with single period T0, time-step dt, 
    datalength (npts-1)*dt in seconds and given amplitude. Shape of envelope
    is exponential-like.
    """

    # simple check for nyquist
    if T0 < 2*dt:
        raise ValueError('T0 should be at least 2*dt')

    # set up the time vector
    t = np.linspace(0, (npts-1)*dt, npts)
    n_vect = np.random.randn(npts)*noise

    # signal = sin(2*pi/T0)
    signal = np.sin(t * 2*np.pi / T0)

    # modulation = exp(-(t-tmid)^2/sigma^2)
    heaviside = 0.5 * (np.sign(t-tstart)+1)
    modulation = amplitude * np.exp(-(t-tstart)/tlen)

    # create obspy trace
    header = {'delta': dt, 'npts': npts, 'station': 'SYN', 'network': 'XX',
              'channel': 'HHZ', 'location': '00'}
    tr = Trace(data=signal*heaviside*modulation+n_vect, header=header)

    # return
    return tr


def generate_synthetic_data(ntraces, data_dir, cat_name):
    """
    Generate ntraces synthetic traces of random types, at random times. Write
    corresponding catalog and data to disk.
    """

    # get trace parameters fom various random distributions
    dt = 0.01
    delta_otimes = poisson.rvs(15.0, size=ntraces)
    trace_type = np.random.randint(2, size=ntraces)
    npts = np.random.randint(1001, 1201, size=ntraces)
    T0 = uniform.rvs(0.1, 0.05, size=ntraces)
    amp = uniform.rvs(1.0, 2.0, size=ntraces)
    noise = uniform.rvs(0.0, 0.1, size=ntraces)
    mu = norm.rvs(5.0, 1.0, size=ntraces)
    sigma = norm.rvs(2.0, 1.0, size=ntraces)
    tstart = norm.rvs(2.0, 1.0, size=ntraces)
    tlen = norm.rvs(1.0, 0.5, size=ntraces)

    f_ = open(join(data_dir, cat_name), 'w')
    f_.write("Otime                  Type\n")
    starttime = UTCDateTime(2015, 1, 1, 0, 0, 0, 0)
    otime = 0.0
    for i in xrange(ntraces):
        otime = otime + delta_otimes[i]
        ttype = trace_type[i]
        if ttype == 0:
            tr = generate_synthetic_gaussian_tremor(3*T0[i], dt, npts[i],
                                                    amp[i], noise[i], mu[i],
                                                    sigma[i])
        else:
            tr = generate_synthetic_exponential_tremor(T0[i], dt, npts[i],
                                                       amp[i], noise[i],
                                                       tstart[i], tlen[i])
    
        # set the start-time correctly
        tr.stats.starttime = starttime + otime
        # write to file
        fname = tr.stats.starttime.isoformat() + '_' + tr.id + ".MSEED"
        print fname
        tr.write(join(data_dir,fname), format="MSEED")
        # write to catalog
        f_.write("%s\t%d\n"%(tr.stats.starttime.isoformat(), ttype))

    f_.close()
    


if __name__ == '__main__':

    import matplotlib.pyplot as plt
    from obspy.signal.filter import envelope
    from scipy.stats import kurtosis

    tr1 = generate_synthetic_gaussian_tremor(0.1, 0.01, 1001, 5.0, 0.1, 5.0, 2.0)
    tr2= generate_synthetic_exponential_tremor(0.1, 0.01, 1001, 5.0, 0.1, 2.0, 2.0)
    env1 = envelope(tr1.data)
    env2 = envelope(tr2.data)

    npts = tr1.stats.npts
    dt = tr1.stats.delta
    t = np.linspace(0, (npts-1)*dt, npts)
    fig, axes = plt.subplots(2, 1)
    plt.sca(axes[0])
    plt.plot(t, tr1.data, 'b', label='Kurtosis = %.2f'%(kurtosis(tr1.data)))
    plt.plot(t, env1, 'r', label='Max/Mean = %.2f'%(np.max(env1)/np.mean(env1)))
    plt.title('Gaussian')
    plt.legend()
    plt.sca(axes[1])
    plt.plot(t, tr2.data, 'b', label='Kurtosis = %.2f'%(kurtosis(tr2.data)))
    plt.plot(t, env2, 'r', label='Max/Mean = %.2f'%(np.max(env2)/np.mean(env2)))
    plt.title('Exponential')
    plt.legend()

    plt.show()

