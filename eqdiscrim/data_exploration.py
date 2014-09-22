import sklearn as sk
import numpy as np
import matplotlib.pyplot as plt
from cat_io.sihex_io import read_sihex_xls
from preproc import latlon_to_xy
from sklearn.preprocessing import StandardScaler


# read catalog
Xlatlon, y, names_latlon = read_sihex_xls()

# extract the x and y coordinates
print Xlatlon.shape, names_latlon
ilat = 2
ilon = 3
X, names = latlon_to_xy(Xlatlon, names_latlon, ilat, ilon)
X_xy = X[:, [ilon, ilat]]

# scale geographical coordinates
scaler = StandardScaler().fit(X_xy)
X_xy = scaler.transform(X_xy)

# plot
plt.scatter(X_xy[:, 0], X_xy[:, 1])
plt.savefig('sismicite_sihex.png')

