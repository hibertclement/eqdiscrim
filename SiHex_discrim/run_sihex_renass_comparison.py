import os
import cat_io.sihex_io as io
import matplotlib.pyplot as plt
from pickle import load, dump

catalog_dir = '../static_catalogs'
tidy_file = 'sihex_tidy_all_dataframe.dat'

read_tidy = True

if read_tidy:

    print "Loading from tidy file"
    # load from file
    with open(os.path.join(catalog_dir, tidy_file), 'rb') as f_:
        df = load(f_)
else:

    # Re-read the sihex files from the start
    df = io.read_all_sihex_files(catalog_dir)
    df = io.clean_sihex_data(df)
    df = io.mask_to_sihex_boundaries(catalog_dir, df)

    # dump to file
    with open(os.path.join(catalog_dir, tidy_file), 'wb') as f_:
        dump(df, f_)

# keep only points within sihex boundaries
sihex_df = df[df['IN_SIHEX']]
print sihex_df['TYPE'].value_counts()
print sihex_df.head()

plt.figure()
sihex_df.plot(kind='scatter', x='LON', y='LAT')
plt.savefig('in_scatter.png')

