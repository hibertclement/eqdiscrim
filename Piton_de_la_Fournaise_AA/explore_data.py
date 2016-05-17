# -*- coding: utf-8 -*-
from pickle import load
from os.path import join
from seis_class import clean_datasets
import numpy as np
import matplotlib.pyplot as plt

figdir = 'figures'
X_filename = 'XRF.dat'
Y_filename = 'YRF.dat'

do_clean = True # if True, there are just Som and Eff and rm np.log10 for the Energy axe
                # if False, add np.log10 for the Energy axe

if do_clean :
    X, att_names, Y, Ydict = clean_datasets(X_filename, Y_filename)
else :
    f_ = open(X_filename, 'r')
    X, att_names = load(f_) 
    f_.close()

    f_ = open(Y_filename, 'r')
    Y, Ydict = load(f_)
    f_.close()

n_ev, n_att = X.shape

#def plot_histograms(X_indexes, hist_range, xlabel, filename):
#
#    plt.savefig(join(figdir, filename))
#    pass

## in progress


# add plotting stuff here :-)

ntypes = len(Ydict)

duration = []
amplitude = []
rapport = []
kurtosis = []
max_mean = []
dominant_f = []
central_f = []
energy = []
AsDec = []

# cycle through the attributes
for i in xrange(n_att):
    if att_names[i] == 'DUR':
        duration.append(i)
    elif att_names[i] == 'A':
        amplitude.append(i)
    elif att_names[i] == 'dur/A':
        rapport.append(i)
    elif att_names[i] == 'K':
        kurtosis.append(i)
    elif att_names[i] == 'maxA_mean':
        max_mean.append(i)
    elif att_names[i] == 'dom_f':
        dominant_f.append(i)
    elif att_names[i] == 'cent_f':
        central_f.append(i)
    elif att_names[i] == 'E':
        energy.append(i)
    elif att_names[i] == 'AsDec':
        AsDec.append(i)
    
# colors
c = ('red', 'blue', 'green', 'orange', 'cyan', 'yellow', 'pink', 'black', 'grey')
    
        
# histograms for all attributs and for FJS ( = the best) in one figure

nl = 3
nc = 3

fig = plt.figure(figsize = (5, 5))
fig, axes = plt.subplots(nl, nc)

fig.subplots_adjust(left = 0.06, bottom = 0.1,
                       right = 0.98, top = 0.8, wspace = 0.4, hspace = 0.5)

plt.subplot(3, 3, 1)
                       
values = X[:, duration[1]]
#plt.sca(axes[0])
for itype in Ydict.keys():
    plt.hist(values[Y==itype], normed=True, bins=20, range=(0, 20),
             color=c[itype], alpha=0.2)

plt.xlabel('Duration (s)')
plt.legend(['727 rockfalls', '2550 Sommitals'], loc = 'upper left', bbox_to_anchor=(0, 1.8))


plt.subplot(3, 3, 2)
values = X[:, amplitude[1]]
for itype in Ydict.keys():
    plt.hist(values[Y==itype], normed=True, bins=20, range=(-0.5, 2.5),
             color=c[itype], alpha=0.2,
             label='%d %s'%(len(values[Y==itype]), Ydict[itype]))
plt.xlabel('log10(Amplitude (um/s))')

plt.subplot(3, 3, 3)
values = X[:, rapport[1]]
for itype in Ydict.keys():
    plt.hist(np.log10(values[Y==itype]), normed=True, bins=20, range=(-3, 2),
             color=c[itype], alpha=0.2,
             label='%d %s'%(len(values[Y==itype]), Ydict[itype]))
plt.xlabel('Rapport (s^2/um)')


plt.subplot(3, 3, 4)
values = X[:, kurtosis[1]]
for itype in Ydict.keys():
    plt.hist(values[Y == itype], normed=True, bins=20, range=(0, 1.5), 
             color=c[itype], alpha=0.2,
             label='%d %s'%(len(values[Y==itype]), Ydict[itype]))
plt.xlabel('log10(Kurtosis)')


plt.subplot(3, 3, 5)
values = X[:, max_mean[1]]
for itype in Ydict.keys():
    plt.hist(values[Y == itype], normed=True, bins=20, range=(2, 10), 
             color=c[itype], alpha=0.2,
             label='%d %s'%(len(values[Y==itype]), Ydict[itype]))
plt.xlabel('maxA/mean of envelope')

plt.subplot(3, 3, 6)
values = X[:, dominant_f[1]]
for itype in Ydict.keys():
    plt.hist(values[Y == itype], normed=True, bins=20, range=(5, 50), 
             color=c[itype], alpha=0.2,
             label='%d %s'%(len(values[Y==itype]), Ydict[itype]))
plt.xlabel('Dominant frequency (hz)')


plt.subplot(3, 3, 7)
values = X[:, central_f[1]]
for itype in Ydict.keys():
    plt.hist(values[Y == itype], normed=True, bins=20, range=(0, 15), 
             color=c[itype], alpha=0.2,
             label='%d %s'%(len(values[Y==itype]), Ydict[itype]))
plt.xlabel('Central frequency (Hz)')


plt.subplot(3, 3, 8)
values = X[:, energy[1]]
for itype in Ydict.keys():
    plt.hist(values[Y == itype], normed=True, bins=20, range=(0, 11), 
             color=c[itype], alpha=0.2,
             label='%d %s'%(len(values[Y==itype]), Ydict[itype]))
plt.xlabel('log10(Energy)')

plt.subplot(3, 3, 9)
values = X[:, AsDec[1]]
for itype in Ydict.keys():
    plt.hist(values[Y == itype], normed=True, bins=20, range=(-1, 7), 
             color=c[itype], alpha=0.2,
             label='%d %s'%(len(values[Y==itype]), Ydict[itype]))
plt.xlabel('AsDec')
    

plt.show()
fig.savefig('histograms.png')
plt.clf()

#plt.figure(figsize = (100 / 72.0, 80 / 72.0)); plt.savefig('myFile.png', dpi = 72)

""" #here plot two histograms per figure
    #one histogram = one station
    #one figure = one attribut

fig, axes = plt.subplots(1, len(duration))
for i in range(len(duration)):
    values = X[:, duration[i]]
    plt.sca(axes[i])
    for itype in Ydict.keys():
        plt.hist(values[Y==itype], normed=True, bins=20, range=(0, 70),
                 color=c[itype], alpha=0.2,
                 label='%d %s'%(len(values[Y==itype]), Ydict[itype]))
    plt.xlabel('Duration (s)')
    plt.legend()
    plt.title('Station %d'%(i+1))

#plt.show()
plt.savefig('duration.png')
plt.clf()

fig, axes = plt.subplots(1, len(amplitude))
for i in range(len(amplitude)):
    values = X[:, amplitude[i]]
    plt.sca(axes[i])
    for itype in Ydict.keys():
        plt.hist(values[Y==itype], normed=True, bins=20, range=(-0.5, 3.5),
                 color=c[itype], alpha=0.2,
                 label='%d %s'%(len(values[Y==itype]), Ydict[itype]))
    plt.xlabel('log10(Amplitude (um/s))')
    plt.legend()
    plt.title('Station %d'%(i+1))

#plt.show()
plt.savefig('amplitude.png')
plt.clf()

fig, axes = plt.subplots(1, len(rapport))
for i in range(len(rapport)):
    values = X[:, rapport[i]]
    plt.sca(axes[i])
    for itype in Ydict.keys():
        plt.hist(np.log10(values[Y==itype]), normed=True, bins=20, range=(-3, 20),
                 color=c[itype], alpha=0.2,
                 label='%d %s'%(len(values[Y==itype]), Ydict[itype]))
    plt.xlabel('Rapport (s^2/um)')
    plt.legend()
    plt.title('Station %d'%(i+1))

#plt.show()
plt.savefig('rapport.png')
plt.clf()

fig, axes = plt.subplots(1, len(kurtosis))
for i in range(len(kurtosis)):
    values = X[:, kurtosis[i]]
    plt.sca(axes[i])
    for itype in Ydict.keys():
        plt.hist(values[Y == itype], normed=True, bins=20, range=(0, 2), 
                 color=c[itype], alpha=0.2,
                 label='%d %s'%(len(values[Y==itype]), Ydict[itype]))
    plt.xlabel('log10(Kurtosis)')
    plt.legend()
    plt.title('Station %d'%(i+1))

#plt.show()
plt.savefig('kurtosis.png')
plt.clf()

fig, axes = plt.subplots(1, 2)
for i in range(len(max_mean)):
    values = X[:, max_mean[i]]
    plt.sca(axes[i])
    for itype in Ydict.keys():
        plt.hist(values[Y == itype], normed=True, bins=20, range=(0, 30), 
                 color=c[itype], alpha=0.2,
                 label='%d %s'%(len(values[Y==itype]), Ydict[itype]))
    plt.xlabel('maxA/mean of envelope')
    plt.legend()
    plt.title('Station %d'%(i+1))

#plt.show()
plt.savefig('maxA_mean.png')
plt.clf()

fig, axes = plt.subplots(1, 2)
for i in range(len(dominant_f)):
    values = X[:, dominant_f[i]]
    plt.sca(axes[i])
    for itype in Ydict.keys():
        plt.hist(values[Y == itype], normed=True, bins=20, range=(6, 50), 
                 color=c[itype], alpha=0.2,
                 label='%d %s'%(len(values[Y==itype]), Ydict[itype]))
    plt.xlabel('Dominant frequency (hz)')
    plt.legend()
    plt.title('Station %d'%(i+1))

#plt.show()
plt.savefig('dominant_f.png')
plt.clf()

fig, axes = plt.subplots(1, 2)
for i in range(len(central_f)):
    values = X[:, central_f[i]]
    plt.sca(axes[i])
    for itype in Ydict.keys():
        plt.hist(values[Y == itype], normed=True, bins=20, range=(0, 15), 
                 color=c[itype], alpha=0.2,
                 label='%d %s'%(len(values[Y==itype]), Ydict[itype]))
    plt.xlabel('Central frequency (Hz)')
    plt.legend()
    plt.title('Station %d'%(i+1))

#plt.show()

plt.savefig('central_f.png')
plt.clf()

fig, axes = plt.subplots(1, 2)
for i in range(len(energy)):
    values = X[:, energy[i]]
    plt.sca(axes[i])
    for itype in Ydict.keys():
        plt.hist(values[Y == itype], normed=True, bins=20, range=(0, 15), 
                 color=c[itype], alpha=0.2,
                 label='%d %s'%(len(values[Y==itype]), Ydict[itype]))
    plt.xlabel('log10(Energy)')
    plt.legend()
    plt.title('Station %d'%(i+1))

#plt.show()

plt.savefig('Energy.png')
plt.clf()

fig, axes = plt.subplots(1, 2)
for i in range(len(AsDec)):
    values = X[:, AsDec[i]]
    plt.sca(axes[i])
    for itype in Ydict.keys():
        plt.hist(values[Y == itype], normed=True, bins=20, range=(-2, 20), 
                 color=c[itype], alpha=0.2,
                 label='%d %s'%(len(values[Y==itype]), Ydict[itype]))
    plt.xlabel('AsDec')
    plt.legend()
    plt.title('Station %d'%(i+1))

#plt.show()

plt.savefig('AsDec.png')
plt.clf()
"""