import glob
import argparse
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from obspy import UTCDateTime, read, Stream, read_inventory
from obspy.io.xseed import Parser
import attributes as att
import eqdiscrim_io as io

pd.set_option('mode.use_inf_as_null', True)

def get_data_OVPF(cfg, starttime, window_length):
    
    inv = read_inventory(cfg.response_fname)
    parser = Parser(cfg.BOR_response_fname)

    st = Stream()
    for sta in cfg.station_names:
        if sta == 'BOR':
            st_tmp = io.get_waveform_data(starttime, window_length,
                                                    'PF', sta, '??Z', parser,
                                                    simulate=True)
        else:
            st_tmp = io.get_waveform_data(starttime, window_length,
                                                    'PF', sta, '??Z', inv)

        if st_tmp is not None:

            st += st_tmp

    return st

def get_clf_dict_from_combinations(combinations, clfdir):

    clf_dict = {}
    for key in combinations:
        fname = os.path.join(clfdir, 'clf_%s.dat' % key)
        clf = io.load(fname)
        clf_dict[key] = clf

    return clf_dict

def create_clf_dict_and_dataframe(clfdir, sta_names):

    clf_fnames = glob.glob(os.path.join(clfdir, 'clf_*.dat'))
    clf_dict = {}
    for fname in clf_fnames:
        sta = os.path.basename(fname).split('_')[1].split('.')[0]
        clf_dict[sta] = fname

    index = clf_dict.keys()
    
    series = {}
    for sta in sta_names:
        series[sta] = pd.Series([sta in ind for ind in index], index=index)
    clf_df = pd.DataFrame(series)

    return clf_dict, clf_df


def get_clf_key_from_stalist(clf_df, sta_list):

    all_stations = clf_df.columns.values

    # get all clfs that contain all the stations
    for i in xrange(len(all_stations)):
        sta = all_stations[i]
        if sta in sta_list:
            if i == 0:
                df = clf_df[clf_df[sta] == True]
            else:
                df = df[df[sta] == True]
        else:
            if i == 0:
                df = clf_df[clf_df[sta] == False]
            else:
                df = df[df[sta] == False]

    return df.index.values[0]


def get_clf_key_from_stalist_and_combinations(sta_list, combinations):
    
    for key in combinations:
        key_stas = key.split('+')
        for sta in key_stas:
            if sta in sta_list:
                found_key = True
            else:
                found_key = False
                break
            # if you get here, then this combination is good
            if found_key:
                break
            
    if found_key:
        return key
    else:
        return None


def plot_prob(prob, classes, starttime, cfg):

    width = 0.75
    ind = np.arange(len(prob))
    plt.figure()
    imax = np.argmax(prob)
    plt.barh(ind, prob, width, color='b', alpha=0.5)
    plt.scatter(prob[imax] * 1.05, ind[imax] + width / 2.,  marker='*',
                color='black', s=250)
    plt.xlabel('Probability')
    plt.yticks(ind + width / 2., classes)
    plt.xlim([0, prob[imax] * 1.2])
    plt.title("%s - %s - %.2f percent" % (starttime.isoformat(), classes[imax],
                                          prob[imax] * 100.))

    fname = '%s_%s_%.2f.png' % (starttime.isoformat(), classes[imax],
                                prob[imax])
    plt.savefig(os.path.join(cfg.figdir, fname))
    plt.show()


def run_predict(args):

    # import pdb; pdb.set_trace()
    import time

    t1 = time.time()
    
    # get configuration
    cfg = io.Config(args.config_file)

    # set up classifiers to use
    clf_dict = get_clf_dict_from_combinations(cfg.combinations, cfg.clfdir)

    t2 = time.time()
    # request data from the stations
    if cfg.do_use_saved_data:
        data_fname = os.path.join(cfg.data_dir, "39160_PF.*.MSEED")
        st = read(data_fname, starttime=args.starttime,
                  endtime=args.starttime + args.duration)
    else:
        st = get_data_OVPF(cfg, args.starttime, args.duration)
    t3 = time.time()

    # select classifier as a function of which stations are present
    sta_names = np.unique([tr.stats.station for tr in st])
    # clf_key = get_clf_key_from_stalist(clf_dict, sta_names)
    clf_key = get_clf_key_from_stalist_and_combinations(sta_names,
                                                        cfg.combinations)
    clf = clf_dict[clf_key]

    # read best attributes file and get the best attributes
    best_atts = io.load(cfg.best_atts_fname)
    best_atts_clf = best_atts[clf_key]

    t4 = time.time()
    # calculate attributes, combine and prune according to classifier
    n_sta = len(sta_names)
    for sta in sta_names:
        st_sta = st.select(station=sta)
        attributes, att_names = att.get_all_single_station_attributes(st_sta)
        df_att_sta = pd.DataFrame(attributes, columns=att_names)

        # if there is only one station, best_atts_clf is simple to parse
        if n_sta == 1:
            X_df = df_att_sta
        else:
            # more than one station, need to add station names to attribtues
            # before deciding to keep them or not
            for a in att_names:
                new_a = '%s_%s' % (sta, a)
                df_att_sta.rename(columns={a: new_a}, inplace=True)

            # if this is the first station just copy the data-frame
            if sta == sta_names[0]:
                X_df = df_att_sta.copy()
            else:
                # join it on to the previous one
                X_df = X_df.join(df_att_sta)
    t5 = time.time()

    # run prediction and output result
    X = X_df[best_atts_clf].values
    y = clf.predict(X)
    p_matrix = clf.predict_proba(X)
    t6 = time.time()

    # do plot and print
    print "Event is %s with probability %0.2f" % (y[0], np.max(p_matrix))
    print "Time for entire process %0.2f" % (t6-t1)
    print "Time for data request %0.2f" % (t3-t2)
    print "Time for attribute calculation %0.2f" % (t5-t4)
    print "Time for prediction %0.2f" % (t6-t5)
    
    plot_prob(p_matrix[0, :], clf.classes_, args.starttime, cfg)


if __name__ == '__main__':

    # set up parser
    cl_parser = argparse.ArgumentParser(description='Launch classification.')
    cl_parser.add_argument('config_file', help='eqdiscrim configuration file')
    cl_parser.add_argument('starttime', type=UTCDateTime,
                           help='Timestamp of first point in window')
    cl_parser.add_argument('duration', help='Window duration in seconds',
                           type=float)

    # Effondrement
    # args = cl_parser.parse_args(['eqdiscrim_VF.cfg', '2016-06-02T20:22:25.60',
    #                              '5.96'])

    # Sommital
    # args = cl_parser.parse_args(['eqdiscrim_VF.cfg', '2016-06-08T20:57:24.54',
    #                             '3.8'])

    # Indetermine
    # args = cl_parser.parse_args(['eqdiscrim_VF.cfg', '2016-06-04T03:48:28.52',
    #                             '11.64'])

    # Local
    # args = cl_parser.parse_args(['eqdiscrim_VF.cfg', '2016-06-04T07:18:20.12',
    #                              '9.4'])

    # parse input
    args = cl_parser.parse_args()

    # run program
    run_predict(args)
