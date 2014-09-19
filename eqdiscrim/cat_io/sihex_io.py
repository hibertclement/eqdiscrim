import pandas as pd
import numpy as np


# files to be read (static for now - ugly but efficient - fix later)
sihex_xls = '../static_catalogs/SIHEXV2-inout-final.xlsx'
sihex_txt = '../static_catalogs/SIHEXV2-catalogue-final.txt'


def read_sihex_xls():

    # read the excel file into pandas data frames
    eq_in = pd.read_excel(sihex_xls, sheetname=0)
    eq_out = pd.read_excel(sihex_xls, sheetname=1)
    # eq_out_felt = pd.read_excel(sihex_xls, sheetname=2)

    # transform the pandas format to numpy format for later use in sklearn
    names = list(["ID", "DATE", "HEURE", "LAT", "LON", "PROF", "Mw", "AUTEUR"])
    X_in = eq_in[names].values
    X_out = eq_out[names].values
    X = np.vstack((X_in, X_out))
    n_ev, n_names = X.shape

    # the current data file does not contain "ke" attributes
    # all the events are "ke"
    y = np.empty(n_ev, dtype='S2')
    y[...] = 'ke'

    return X, y, names
