import numpy as np
import matplotlib.pyplot as plt
from cat_io.renass_io import read_stations_fr
from preproc import n_stations_per_year

start_year = 1962
end_year = 2009

# read station catalog
S, names = read_stations_fr()

# save UTC start and end times of stations
istart = 4
iend = 5
S_times = S[:, istart:iend+1]

# get number of stations per year
year_count = n_stations_per_year(S_times, start_year, end_year)

# seem to have four "epochs" : <1970, 1970-1978, 1978-1990, >1990
# take mean and std of the three epochs
ep1 = year_count[year_count[:, 0] < 1970, 1]
ep2 = year_count[(year_count[:, 0] >= 1970) & (year_count[:, 0] < 1978), 1]
ep3 = year_count[(year_count[:, 0] >= 1978) & (year_count[:, 0] < 1990), 1]
ep4 = year_count[year_count[:, 0] >= 1990, 1]

# find median station number
m1 = np.median(ep1)
m2 = np.median(ep2)
m3 = np.median(ep3)
m4 = np.median(ep4)

print m1, m2, m3, m4

# plot
plt.figure()
plt.plot(year_count[:, 0], year_count[:, 1], label='no of stations')
plt.plot((start_year, 1970), (m1, m1), 'r', lw=3, label='median')
plt.text(start_year, m1+5, '%d'%np.int(m1))
plt.plot((1970, 1978), (m2, m2), 'r', lw=3)
plt.text(1970, m2+5, '%d'%np.int(m2))
plt.plot((1978, 1990), (m3, m3), 'r', lw=3)
plt.text(1978, m3+5, '%d'%np.int(m3))
plt.plot((1990, end_year), (m4, m4), 'r', lw=3)
plt.text(1990, m4+5, '%d'%np.int(m4))
plt.xlabel('Year')
plt.ylabel('Number of stations')
plt.title('Epochs by station numbers')
plt.legend(loc='lower right')
plt.savefig('../figures/num_stations_per_year.png')


