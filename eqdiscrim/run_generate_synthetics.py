from synthetics.gen_syn_data import generate_synthetic_data

syn_dir = '../synthetic_data'
cat_name = 'syn_catalog.txt'

generate_synthetic_data(500, syn_dir, cat_name)
