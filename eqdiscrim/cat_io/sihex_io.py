import pandas as pd 
import numpy as np

# files to be read (static for now - ugly but efficient - fix later)
sihex_xls = '../../static_catalogs/SIHEXV2-inout-final.xlsx'
sihex_txt = '../../static_catalogs/SIHEXV2-catalogue-final.txt'

# read the excel file into pandas data frames
eq_in = pd.read_excel(sihex_xls, sheetname=0)
eq_out = pd.read_excel(sihex_xls, sheetname=1)
eq_out_felt = pd.read_excel(sihex_xls, sheetname=2)

# transform the pandas format to numpy format for later use in sklearn
X_in = eq_in[list(["ID", "DATE", "HEURE", "LAT", "LON", "PROF", "Mw",
                   "AUTEUR"])].values
X_out = eq_out[list(["ID", "DATE", "HEURE", "LAT", "LON", "PROF", "Mw",
                     "AUTEUR"])].values

print X_in.shape, X_out.shape

X = np.vstack((X_in, X_out))
print X.shape
