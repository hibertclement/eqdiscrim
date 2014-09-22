import numpy as np
from pyproj import Proj


p = Proj(proj='utm', zone=31, ellps='WGS84')


def latlon_to_xy(X, names, ilat, ilon):
    """
        Given an attribute matrix X, and the indexes corresponding to the
        latitude and longitude attributes, returns a new attribute matrix where
        the lat and lons are now y and x in the appropriate reference system.
    """
    # extract the longitudes and latitudes
    nev, nat = X.shape
    lon = X[:, ilon]
    lat = X[:, ilat]

    # do the projection
    x, y = p(lon, lat)

    # put the results back into the matrix
    X_new = X
    # longitude becomes x
    X_new[:, ilon] = x[:]
    # latitude becomes y
    X_new[:, ilat] = y[:]

    # fix up the names, too
    names_new = names
    names_new[ilat] = 'Y'
    names_new[ilon] = 'X'

    return X_new, names_new

def xy_to_latlon(X, names, ix, iy):
    """
        Given an attribute matrix X, and the indexes corresponding to the
        x and y attributes, returns a new attribute matrix where
        the x and y are now lon and lat in the appropriate reference system.
    """
    # extract the longitudes and latitudes
    nev, nat = X.shape
    x = X[:, ix]
    y = X[:, iy]

    # do the projection
    lon, lat = p(x, y, inverse=True)

    # put the results back into the matrix
    X_new = X
    # x becomes longitude 
    X_new[:, ix] = lon[:]
    # y becomes latitude 
    X_new[:, iy] = lat[:]

    # fix up the names, too
    names_new = names
    names_new[ix] = 'LAT'
    names_new[iy] = 'LON'

    return X_new, names_new
