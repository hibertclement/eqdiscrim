import glob
import os
import pickle
import eqdiscrim_io as io

station_names = ["RVL", "FLR", "BOR", "BON", "SNE", "FJS", "CSS", "GPS", "GPN", "FOR"]
att_dir_in = "Attributes"
att_dir_out = "Attributes_grouped"
group_size = 500

if not os.path.exists(att_dir_out):
    os.mkdir(att_dir_out)

for sta in station_names:

    print "Treating station %s" % sta

    # read and cat all relevant dataframes
    fnames = glob.glob(os.path.join(att_dir_in, 'X_%s_*_dataframe.dat' % (sta)))
    if len(fnames) == 0:
        continue
    print "Reading and concatenating %d dataframes" % len(fnames)
    X_df_full = io.read_and_cat_dataframes(fnames)

    print "Writing grouped dataframes"
    n_events = len(X_df_full)
    start_i = 0
    while start_i < n_events:
        fname = os.path.join(att_dir_out, 'X_%s_%05d_dataframe.dat' % (sta, start_i))
        end_i = min(n_events-1, start_i + group_size)
        df = X_df_full[start_i : end_i]
        print len(df)
        f_ = open(fname, 'w')
        pickle.dump(df, f_)
        f_.close()
        start_i = start_i + group_size
