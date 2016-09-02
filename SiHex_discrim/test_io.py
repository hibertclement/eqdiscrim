import pytest
import numpy as np
from shapely.geometry import Point, Polygon
from sihex_io import read_all_sihex_files
from sihex_io import mask_to_sihex_boundaries
from renass_io import read_stations_fr_dataframe
from preproc import add_distance_to_closest_stations
from dateutil import tz

utc = tz.gettz('UTC')


def test_polygon():

    # make an irregular polygon cricumscribable by a circle
    npts = 100
    r = 5
    angles = np.random.rand(npts)
    angles.sort()
    angles = angles*2*np.pi
    x = r * np.cos(angles)
    y = r * np.sin(angles)
    xy_tup = zip(x, y)
    poly = Polygon(xy_tup)

    p1 = Point(r*0.2, r*0.2)
    p2 = Point(r*1.5, r*1.5)

    assert (p1.within(poly))
    assert not (poly.contains(p2))


@pytest.fixture()
def sihex_df(request):
    return read_all_sihex_files('test_data')

def test_read_sihex(sihex_df):
    
    nlines, ncol = sihex_df.shape

    assert nlines == 300
    assert ncol == 7

def test_mask_sihex(sihex_df):

    mask_to_sihex_boundaries('test_data', sihex_df)
    assert True

def test_read_stations():

    sta_df = read_stations_fr_dataframe()

    nsta_new, natt_new = sta_df.shape
    assert nsta_new == 306
    assert natt_new == 6

def test_distance_closest_stations(sihex_df):
    
    nr_orig, nc_orig = sihex_df.shape

    df = add_distance_to_closest_stations(sihex_df, 3)
    nr, nc = df.shape

    assert nr == nr_orig
    assert nc == nc_orig + 3
    

if __name__ == '__main__':

    unittest.TextTestRunner(verbosity=2).run(suite())
