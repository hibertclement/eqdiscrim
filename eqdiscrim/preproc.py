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


def dist_to_n_closest_stations(X_xy, S_xy, n, timing=False):
    """
    Returns a numpy array containing the distance (in reduced coordinates) to
    the n nearest stations. If there are fewer than n stations, then the
    returned array is padded with the largest available distance for each
    event. If no stations are available, the event is ignored.

    :param X_xy: a 2D numpy array containing one line per event, and whose
        first two columns contain x and y coordinates of the event
    :param S_xy: a 2D numpy array containing one line per station, and whose
        first two columns contain x and y coordinates of the station
    :param n: number of nearest stations to consider
    :type timing: boolean, optional
    :param timing: if True, then X_xy contains a third column with the orgin
        time as a datetime object, and X_xy contains a third and fourth column
        with start and end times of the station as datetime objects
    :rtype: a 2D numpy array containing one line per event and n columns, the
        distances to the n closest stations. This array may contain fewer
        events than the input X_xy array
    """

    nev, nd = X_xy.shape
    nst, nd = S_xy.shape

    dist = np.empty((nev, n), dtype=np.float)

    idist = 0   # need an independent counter for idist
    for iev in xrange(nev):
        xev = X_xy[iev, 0]
        yev = X_xy[iev, 1]
        if timing:
            tev = X_xy[iev, 2]
            # get the distance to each point taking timing into account
            d = np.array([np.sqrt((xev-S_xy[ist, 0])**2+(yev-S_xy[ist, 1])**2)
                          for ist in xrange(nst) if (tev >= S_xy[ist, 2] and
                                                     tev <= S_xy[ist, 3])])
        else:
            # get the distance to each point
            d = np.array([np.sqrt((xev-S_xy[ist, 0])**2+(yev-S_xy[ist, 1])**2)
                          for ist in xrange(nst)])
        # sort in ascending order
        d.sort()
        # save the n distances
        try:
            dist[idist, 0:n] = d[0:n]
            idist = idist + 1
        except ValueError:
            nd = len(d)
            if nd == 0:
                # the event occurred when there were no stations to record it
                # this case should never happen, but if it does, the event
                # should be ignored (do not increasce idist counter)
                pass
            else:
                # there are fewer than n stations
                # pad the distance matrix with the largest available distance
                dist[idist, 0:nd] = d[0:nd]
                dist[idist, nd:n] = d[nd-1]
                idist = idist + 1

    # resize array if necessary
    dist.resize((idist, n))

    return dist


def n_stations_per_year(S_start_end, start_year, end_year):
    """
    Returns the number of stations per year (only precise to the nearest year).

    :param S_start_end: a 2D ndarray containing one row per station and two
        columns containing respectively the start and end times of the station
        as datetime objects
    :param start_year: first year of interest
    :param end_year: last year of interest
    :rtype: a 2D ndarray containing years and number of active stations in its
        two columns
    """

    nst, nd = S_start_end.shape
    nyears = end_year - start_year + 1

    year_count = np.empty((nyears, 2), dtype=np.int)

    for iy in xrange(nyears):
        year = start_year + iy
        sta = [i for i in xrange(nst)
               if (year >= S_start_end[i, 0].year and
                   year <= S_start_end[i, 1].year)]
        nsta = len(sta)
        year_count[iy, 0] = year
        year_count[iy, 1] = nsta

    return year_count


def GutenbergRichter(magnitudes, mag_min, mag_max, step):
    """
    Computes the Gutenberg-Richter law from a list of magnitudes.
    """

    mags = np.arange(mag_min, mag_max, step)
    N, mags = np.histogram(magnitudes, mags)
    nmags = len(N)
    log10N = np.zeros(nmags, dtype=np.float)
    for i in xrange(nmags):
        try:
            log10N[i] = np.log10(np.sum(N[i:nmags]))
        except RuntimeWarning:
            log10N[i] = 0.0

    return log10N, mags


def toHourFraction(date):
    """
    Computes fractional hours
    """

    s = date.minute*60. + date.second + date.microsecond/1.e6
    hour_fraction = s / 3600.

    return date.hour + hour_fraction


def toWeekdayFraction(date):
    """
    Computes fractional weekday
    """

    s = date.hour*3600. + date.minute*60. + date.second + date.microsecond/1.e6
    day_fraction = s / (3600.*24.)

    return date.isoweekday() + day_fraction
