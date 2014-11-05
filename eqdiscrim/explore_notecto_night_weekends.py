import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np


# ## Exploring non-tectonic events that occur at night or at the weekends
# 
# Some non-tectonic events from the SiHex catalogue seem to occur at night and
# at weekends (see figures that were created using the
# **explore_detectability.py** script). The big question is why...

# ### Step 1 : extract night and weekend events
# 
# The night-time events and the weekend non-tectonic events can easily be
# extracted from the full catalogs. For now, let's just see what types they
# have...

from cat_io.sihex_io import read_sihex_tidy
Btable = '../static_catalogs/sihex_tidy_blasts.dat'
Hfile = '../static_catalogs/sihex_tidy_header.txt'

B, coldict = read_sihex_tidy(Btable, Hfile)
B_hour = B[:, coldict['LocalHour']]
B_weekday = B[:, coldict['LocalWeekday']]
B_night = B[:, :][(B_hour < 8) | (B_hour > 18)]
B_weekend = B[:, :][B_weekday > 5]


nev, nd = B.shape
nev_night, nd = B_night.shape
nev_weekend, nd = B_weekend.shape
print "Of %d non-tectonic events, %d (i.e. %.2f%%) occur at night and %d (i.e. %.2f%%) occur at weekends." %  (nev, nev_night, nev_night/np.float(nev) * 100, nev_weekend, nev_weekend/np.float(nev) *100 )


# In[107]:

#common_ids = np.intersect1d(B_night[:, coldict['ID']], B_weekend[:, coldict['ID']])
common_ids = np.union1d(B_night[:, coldict['ID']], B_weekend[:, coldict['ID']])
n_common = len(common_ids)
print "%d (i.e. %.2f%%) of non-tectonic events occur during the night or at weekends." %    (n_common, n_common/np.float(nev)*100)
# get the common events
B_night_weekend = np.empty((n_common, nd), dtype=object)
for i in xrange(n_common):
    B_night_weekend[i, :] = B[:, :][B[:, coldict['ID']] == common_ids[i]]

all_types = np.unique(B[:, coldict['Type']])
n_types = {}
n_types_night_weekend = {}
for t in all_types:
    n_types_night_weekend[t] = np.sum([B_night_weekend[:, coldict['Type']] == t])
    n_types[t] = np.sum([B[:, coldict['Type']] == t])


# In[109]:

labels = n_types.keys()
values = n_types.values()
labels_night_weekend = n_types_night_weekend.keys()
values_night_weekend = n_types_night_weekend.values()
plt.figure()
fig, axes = plt.subplots(1, 2)
fig.set_size_inches(10, 5)
plt.sca(axes[0])
plt.pie(values, labels=labels, autopct='%d%%')
plt.title('All non-tectonic events')
plt.sca(axes[1])
plt.pie(values_night_weekend, labels=labels_night_weekend, autopct='%d%%')
plt.title('Night or weekend non-tectonic events')
plt.show()

n_types_night_weekend


# In[110]:



