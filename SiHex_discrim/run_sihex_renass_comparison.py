import cat_io.sihex_io as io

catalog_dir = '../static_catalogs'

df = io.read_all_sihex_files(catalog_dir)
df = io.clean_sihex_data(df)
df = io.mask_to_sihex_boundaries(catalog_dir, df)

print df['TYPE'].value_counts()
