# -*- coding: utf-8 -*-
# carte réseau sismique (IN PROGRESS)


from mpl_toolkits.basemap import Basemap
import numpy as np
import matplotlib.pyplot as plt

# llcrnrlat,llcrnrlon,urcrnrlat,urcrnrlon
# are the lat/lon values of the lower left and upper right corners
# of the map.
# lat_ts is the latitude of true scale.
# resolution = 'c' means use crude resolution coastlines.
m = Basemap(projection='merc',llcrnrlat=-22,urcrnrlat=-20,\
            llcrnrlon=50,urcrnrlon=60.5,resolution='f')
m.drawcoastlines()
m.fillcontinents(color='coral',lake_color='aqua')
# draw parallels and meridians.
#m.drawparallels(np.arange(-90.,91.,30.))
#m.drawmeridians(np.arange(-180.,181.,60.))
#m.drawmapboundary(fill_color='aqua')
plt.title("Mercator Projection")
plt.show()



"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap

airports = np.genfromtxt("airports.dat",
                         delimiter=',', 
                         dtype=[('lat', np.float32), ('lon', np.float32)], 
                         usecols=(6, 7))

fig = plt.figure()

themap = Basemap(projection='gall',
              llcrnrlon = -15,              # lower-left corner longitude
              llcrnrlat = 28,               # lower-left corner latitude
              urcrnrlon = 45,               # upper-right corner longitude
              urcrnrlat = 73,               # upper-right corner latitude
              resolution = 'l',
              area_thresh = 100000.0,
              )
              

themap.drawcoastlines()
themap.drawcountries()
themap.fillcontinents(color = 'gainsboro')
themap.drawmapboundary(fill_color='steelblue')

x, y = themap(airports['lon'], airports['lat'])
themap.plot(x, y, 
            'o',                    # marker shape
            color='Indigo',         # marker colour
            markersize=4            # marker size
            )

plt.show()

"""