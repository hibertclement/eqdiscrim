import pandas as pd
import numpy as np


# files to be read (static for now - ugly but efficient)
renass_txt = '../static_catalogs/RENASS_stations_2014.txt'
stations_csv = '../static_catalogs/stations_fr.csv'

def read_renass():

    # read the txt file using pandas
    s_pd = pd.read_table(renass_txt)

    names = list(["Code", "Lat", "Lon", "Elev(m)"])
    stations = s_pd[names].values

    return stations, names

def read_stations_fr():

    # read the csv file using pandas
    s_pd = pd.read_csv(stations_csv)

    names = list(["NOM", "LATITUDE", "LONGITUDE", "ELEV"])
    names_DT = list(["DATEDEB", "HDEB", "DATEFIN", "HFIN"])

    stations = s_pd[names].values
    return stations, names

