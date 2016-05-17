import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from shapely.geometry import Polygon, Point

# load matrix
mat_contents = loadmat('matrice.mat')
X = mat_contents['Mat']
X_att = ['ID', 'LocalHour', 'Latitude', 'Longitude', 'Weekday']
npts, natt = X.shape

# load sihex boundary file
sihex_bound = '../../static_catalogs/line20km.xy.txt'
sh_names = list(["LON", "LAT"])
sh_tmp = pd.read_table(sihex_bound, sep='\s+', header=None, names=sh_names)
sh = sh_tmp[sh_names].values

# load coastline file and corsica file
coast_tmp = pd.read_table('coastline.txt', sep='\s+', header=None, names=sh_names)
coast = coast_tmp[sh_names].values
coast_tup = zip(coast[:,0], coast[:,1])
coast_poly = Polygon(coast_tup)

corsica_tmp = pd.read_table('corsica.txt', sep='\s+', header=None, names=sh_names)
corsica = corsica_tmp[sh_names].values
corsica_tup = zip(corsica[:,0], corsica[:,1])
corsica_poly = Polygon(corsica_tup)

# create polygons for four quadrants
nw_poly = Polygon([(-11.0, 46.0), (3.0, 46.0), (3.0, 52.0), (-11.0, 52.0), (-11.0, 46.0)])
ne_poly = Polygon([(3.0, 46.0), (11.0, 46.0), (11.0, 52.0), (3.0, 52.0), (3.0, 46.0)])
se_poly = Polygon([(3.0, 40.0), (11.0, 40.0), (11.0, 46.0), (3.0, 46.0), (3.0, 40.0)])
sw_poly = Polygon([(-11.0, 40.0), (3.0, 40.0), (3.0, 46.0), (-11.0, 46.0), (-11.0, 40.0)])

# cut sihex

# plot_earthquakes
lat = X[:, 2]
lon = X[:, 3]


# Extract the points that are in the sea
isea = []
iland = []
isea_nw = []
isea_ne = []
isea_se = []
isea_sw = []
iland_nw = []
iland_ne = []
iland_se = []
iland_sw = []
for i in xrange(npts):
    ip = Point(lon[i], lat[i])
    if ip.within(coast_poly) or ip.within(corsica_poly):
        iland.append(i)
        if ip.within(nw_poly):
            iland_nw.append(i)
        elif ip.within(ne_poly):
            iland_ne.append(i)
        elif ip.within(se_poly):
            iland_se.append(i)
        else:
            iland_sw.append(i)
    else:
        isea.append(i)
        if ip.within(nw_poly):
            isea_nw.append(i)
        elif ip.within(ne_poly):
            isea_ne.append(i)
        elif ip.within(se_poly):
            isea_se.append(i)
        else:
            isea_sw.append(i)


plt.figure()
plt.plot(coast[:,0], coast[:,1])
plt.plot(corsica[:,0], corsica[:,1], 'b')
plt.plot(sh[:,0], sh[:,1], 'r')
plt.plot(lon[isea], lat[isea], '.b')
plt.plot(lon[iland], lat[iland], '.g')
plt.savefig('map.png')
