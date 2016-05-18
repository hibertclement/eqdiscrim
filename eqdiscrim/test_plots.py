import synthetics as syn
import attributes as att
import signal_proc as sp
import matplotlib.pyplot as plt
import numpy as np
from copy import deepcopy
from obspy.signal.filter import envelope
from scipy.signal import spline_filter


#sig1 = syn.gen_gaussian_noise(0.01, 54321, 100)
dt = 0.01
npts = 3000
t_vect = np.arange(npts) * dt

sig1 = syn.gen_sweep_poly(dt, npts, [0.05, -0.75, 0.1, 1.0])
sig2 = sig1.copy()

sig2, modulation = syn.modulate_trace_triangle(sig2, 0.2, 300)
env = envelope(sig2.data)
smooth_env1 = sp.smooth(env, window='hanning')
smooth_env2 = sp.smooth(env, window='flat')
assert(len(env) == len(smooth_env1))

plt.plot(t_vect, sig2.data)
plt.plot(t_vect, env)
plt.plot(t_vect, smooth_env1)
plt.plot(t_vect, smooth_env2)

plt.show()

