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

def dist_to_n_closest_stations(X_xy, S_xy, n):
    """
    Returns a numpy array containing the distance (in reduced coordinates) to
    the nth nearest station
    """

    nev, nd = X_xy.shape
    nst, nd = S_xy.shape

    dist = np.empty((nev,n), dtype=np.float)

    for iev in xrange(nev):
        xev = X_xy[iev, 0]
        yev = X_xy[iev, 1]
        # get the distance to each point
        d = np.array([np.sqrt((xev - S_xy[ist, 0])**2 + (yev - S_xy[ist, 1])**2)
                      for ist in xrange(nst)])
        # sort in ascending order
        d.sort() 
        # save the nth value
        dist[iev, 0:n] = d[0:n]

    return dist
