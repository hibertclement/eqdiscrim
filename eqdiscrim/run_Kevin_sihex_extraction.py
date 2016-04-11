import cat_io.sihex_io as io

fname_sihex_tidy = "../static_catalogs/sihex_tidy_all.dat"
fname_sihex_header = "../static_catalogs/sihex_tidy_header.txt"
fname_sihex_Teddy_excel = "../static_catalogs/sihex_tidy_Kevin.xlsx"

# read the full dictionary
X, Xdict = io.read_sihex_tidy(fname_sihex_tidy, fname_sihex_header)

# get a subset of data
header_list = list(['ID', 'X', 'Y', 'Mw', 'LocalHour', 'LocalWeekday', 'Type'])
i_list = [Xdict[name] for name in header_list]
i_otime = Xdict['OriginTime']

nr, nc = X.shape

X_basic = X[:, i_list]
otimes = [X[i, i_otime].isoformat() for i in xrange(nr)]

# write out this subset to an excel file
io.write_sihex_tidy_excel(X[:, i_list], header_list, otimes, fname_sihex_Teddy_excel)

