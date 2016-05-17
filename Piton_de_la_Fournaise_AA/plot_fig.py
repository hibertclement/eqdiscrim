from matplotlib import pyplot as plt
import numpy as np
import collections
from math import *
"""
We plot the linear crossvalidation result for each stations, and the accuracy rate.
2 attributs, linear, cat_global, BOR, RVL, FJS, GPS, SNE
"""
stations = ['BOR', 'RVL', 'FJS', 'GPS', 'SNE']
### data 
# BOR_ar for accuracy rate, BOR_cv for cross-validation, BOR_cvm for the variation of the cv
# BOR_cvaddm for the higher limit of the cross-validation value
# BOR_cvsubm for the lower limit of the cross-validation value

#abscisse : 28 values because there are 28 couple of 2 attributes
x = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28)

BOR_ar = [0.88, 0.90, 0.86, 0.90, 0.89, 0.88, 0.83, 0.79, 0.80, 0.88, 0.79, 0.83, 0.87, 0.73, 0.74, 0.73, 0.71, 0.75, 0.67, 0.71, 0.67, 0.71, 0.64, 0.72, 0.56, 0.63, 0.62, 0.67]
BOR_cv = [0.879, 0.894, 0.891, 0.864, 0.876, 0.892, 0.807, 0.810, 0.770, 0.884, 0.769, 0.837, 0.862, 0.725, 0.746, 0.724, 0.695, 0.749, 0.669, 0.685, 0.671, 0.692, 0.604, 0.715, 0.588, 0.635, 0.576, 0.668]
BOR_cvm = [0.019, 0.014, 0.019, 0.021, 0.032, 0.031, 0.030, 0.018, 0.025, 0.018, 0.044, 0.020, 0.027, 0.021, 0.044, 0.032, 0.020, 0.012, 0.030, 0.029, 0.048, 0.034, 0.029, 0.019, 0.031, 0.022, 0.032, 0.016]

RVL_ar = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
RVL_cv = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
RVL_cvm = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

FJS_ar = [0.90, 0.94, 0.91, 0.90, 0.90, 0.94, 0.85, 0.87, 0.82, 0.95, 0.87, 0.86, 0.90, 0.67, 0.79, 0.70, 0.70, 0.68, 0.79, 0.83, 0.79, 0.83, 0.61, 0.79, 0.63, 0.72, 0.59, 0.69]
FJS_cv = (0.907, 0.931, 0.899, 0.897, 0.896, 0.936, 0.879, 0.871, 0.818, 0.939, 0.890, 0.853, 0.914, 0.681, 0.813, 0.676, 0.696, 0.664, 0.821, 0.801, 0.817, 0.816, 0.617, 0.763, 0.636, 0.704, 0.625, 0.700)
FJS_cvm = (0.011, 0.009, 0.016, 0.020, 0.027, 0.019, 0.022, 0.014, 0.013, 0.014, 0.024, 0.019, 0.022, 0.031, 0.029, 0.029, 0.026, 0.036, 0.020, 0.032, 0.028, 0.037, 0.048, 0.028, 0.051, 0.032, 0.014, 0.037)

GPS_ar = [0.76, 0.83, 0.76, 0.73, 0.74, 0.82, 0.73, 0.72, 0.74, 0.85, 0.82, 0.72, 0.76, 0.58, 0.73, 0.52, 0.55, 0.63, 0.71, 0.76, 0.78, 0.74, 0.56, 0.62, 0.66, 0.57, 0.63, 0.67]
GPS_cv = [0.752, 0.839, 0.753, 0.746, 0.798, 0.840, 0.758, 0.717, 0.741, 0.833, 0.805, 0.730, 0.784, 0.503, 0.737, 0.514, 0.569, 0.651, 0.709, 0.752, 0.725, 0.719, 0.553, 0.604, 0.649, 0.562, 0.644, 0.636]
GPS_cvm = [0.016, 0.02, 0.03, 0.029, 0.019, 0.026, 0.024, 0.022, 0.02, 0.04, 0.019, 0.032, 0.032, 0.043, 0.00, 0.03, 0.032, 0.033, 0.02, 0.033, 0.029, 0.042, 0.0, 0.03, 0.039, 0.012, 0.031, 0.01]

SNE_ar= [0.83, 0.83, 0.83, 0.81, 0.82, 0.88, 0.78, 0.78, 0.79, 0.79, 0.78, 0.79, 0.85, 0.63, 0.69, 0.61, 0.62, 0.64, 0.69, 0.69, 0.70, 0.70, 0.60, 0.59, 0.63, 0.61, 0.56, 0.61]
SNE_cv =  [0.829, 0.833, 0.809, 0.850, 0.834, 0.874, 0.774, 0.772, 0.770, 0.808, 0.777, 0.787, 0.827, 0.609, 0.682, 0.598, 0.614, 0.639, 0.684, 0.683, 0.692, 0.702, 0.580, 0.613, 0.615, 0.596, 0.554, 0.628]
SNE_cvm = [0.026, 0.011, 0.039, 0.007, 0.039, 0.015, 0.027, 0.022, 0.040, 0.014, 0.015, 0.033, 0.032, 0.018, 0.016, 0.021, 0.039, 0.025, 0.033, 0.038, 0.035, 0.036, 0.033, 0.039, 0.036, 0.014, 0.026, 0.025]

# dictionary for the comparison plot

sta_dict = dict(zip(['BOR_ar', 'BOR_cv', 'BOR_cvm', 'RVL_ar', 'RVL_cv', 'RVL_cvm','FJS_ar', 'FJS_cv', 'FJS_cvm','GPS_ar', 'GPS_cv', 'GPS_cvm', 'SNE_ar', 'SNE_cv', 'SNE_cvm'], [BOR_ar, BOR_cv, BOR_cvm, RVL_ar, RVL_cv, RVL_cvm, FJS_ar, FJS_cv, FJS_cvm, GPS_ar, GPS_cv, GPS_cvm, SNE_ar, SNE_cv, SNE_cvm]))
ar_dict = dict(zip(['BOR_ar', 'RVL_ar', 'FJS_ar','GPS_ar',  'SNE_ar'], [BOR_ar, RVL_ar, FJS_ar, GPS_ar,  SNE_ar]))
cv_dict = dict(zip(['BOR_cv', 'RVL_cv', 'FJS_cv', 'GPS_cv', 'SNE_cv'], [BOR_cv,RVL_cv,  FJS_cv,  GPS_cv, SNE_cv]))
cvm_dict = dict(zip(['BOR_cvm', 'RVL_cvm', 'FJS_cvm', 'GPS_cvm', 'SNE_cvm'], [ BOR_cvm,  RVL_cvm,  FJS_cvm, GPS_cvm, SNE_cvm]))

#ordered dictionary
sta_dict = collections.OrderedDict(sorted(sta_dict.items(), key=lambda t: t[0]))
ar_dict = collections.OrderedDict(sorted(ar_dict.items(), key=lambda t: t[0]))
cv_dict = collections.OrderedDict(sorted(cv_dict.items(), key=lambda t: t[0]))
cvm_dict = collections.OrderedDict(sorted(cvm_dict.items(), key=lambda t: t[0]))


keys = sta_dict.keys()
# keys = ['BOR_ar', 'BOR_cv', 'BOR_cvm', 'FJS_ar', 'FJS_cv', 'FJS_cvm', 'GPS_ar', 'GPS_cv', 'GPS_cvm', 'RVL_ar', 'RVL_cv', 'RVL_cvm', 'SNE_ar', 'SNE_cv', 'SNE_cvm']
#sta_dict[keys[0]] = [0.88, 0.9, 0.86, 0.9, 0.89, 0.88, 0.83, 0.79, 0.8, 0.88, 0.79, 0.83, 0.87, 0.73, 0.74, 0.73, 0.71, 0.75, 0.67, 0.71, 0.67, 0.71, 0.64, 0.72, 0.56, 0.63, 0.62, 0.67]
# sta_dict[keys[0]][0]  = 0.88

cv_keys = cv_dict.keys() # BOR FJS GPS RVL SNE (len = 5)
cv_value = cv_dict[cv_keys[0]] # all of the value for BOR_cv (len = 28)

#function for the comparison plot

def stat_mean(sample):
    nev = len(sample)
    mean = np.sum(sample) / nev
    return mean

# data for the comparison plot 
MEAN = []
before_mean = []
for i in xrange(len (cv_value)):
    for j in xrange(len(cv_keys)):
        a = cv_dict[cv_keys[j]][i] # take the first value of cv for each station
        before_mean.append(a) # put them in mean
    
    MEAN.append(stat_mean(before_mean)) # does the mean and put it in the list
    # WARNING = MEAN value are not rounded.
    before_mean = []
    

# plot
plt.figure(1)
#fig = plt.figure(1, figsize = (10, 10))
#plt.subplot(3,2,1)
plt.errorbar(x, BOR_cv, yerr=BOR_cvm, marker='o')
plt.plot(x, BOR_ar, 'ro')
plt.legend(['cv', 'ar'], loc = 'lower left')
plt.title('BOR')
plt.ylim(0.4,1)

"""
plt.subplot(3,2,2)
plt.errorbar(x, RVL_cv, yerr=RVL_cvm, marker='o')
plt.plot(x, RVL_ar, 'wo')
plt.legend(['cv', 'ar'], loc = 'lower left')
plt.title('RVL')
plt.ylim(0,1)

plt.subplot(3,2,3)
plt.errorbar(x, FJS_cv, yerr=FJS_cvm, marker='o')
plt.plot(x, FJS_ar, 'wo')
plt.legend(['cv', 'ar'], loc = 'lower right')
plt.title('FJS')
plt.ylim(0.4,1)


plt.subplot(3,2,4)
plt.errorbar(x, GPS_cv, yerr=GPS_cvm, marker='o')
plt.plot(x, GPS_ar, 'wo')
plt.legend(['cv', 'ar'], loc = 'lower left')
plt.title('GPS')
plt.ylim(0.4,1)

plt.subplot(3,2,5)
plt.errorbar(x, SNE_cv, yerr=SNE_cvm, marker='o')
plt.plot(x, SNE_ar, 'wo')
plt.legend(['cv', 'ar'], loc = 'lower left')
plt.title('SNE')
plt.ylim(0.4,1)
plt.savefig('result.png');

# comparison plot

plt.subplot(3,2,6)
plt.plot(x, MEAN, 'c*')
plt.ylim(0.4,1)
plt.show()
plt.close
"""
# plot to compare BOR and FJS 
fig = plt.figure(2)
plt.errorbar(x, BOR_cv, yerr=BOR_cvm, marker='o')
plt.errorbar(x, FJS_cv, yerr=FJS_cvm, marker='o', color = 'r')
plt.errorbar(x, GPS_cv, yerr=GPS_cvm, marker='o', color = 'c')
plt.errorbar(x, SNE_cv, yerr=SNE_cvm, marker='o', color = 'g')
plt.title('BOR (blue) FJS (red) GPS (cyan) SNE (green)')
plt.ylim(0.4,1)
plt.show()
fig.savefig('cv_result.png')
plt.clf()