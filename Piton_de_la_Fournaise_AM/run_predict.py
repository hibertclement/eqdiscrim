import pickle
import argparse
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from obspy import UTCDateTime, read
import attributes as att
import eqdiscrim_io as io

pd.set_option('mode.use_inf_as_null', True)

def clf_dict_to_sta_dataframe(clf_fname, sta_names):

    f_ = open(clf_fname, 'r')
    clf_dict = pickle.load(f_)
    f_.close()

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
    
def plot_prob(prob, classes, starttime):

    width = 0.75
    ind = np.arange(len(prob))
    fig = plt.figure()
    imax = np.argmax(prob)
    plt.barh(ind, prob, width, color='b', alpha=0.5)
    plt.scatter(prob[imax] * 1.05, ind[imax] + width/2,  marker='*', color='black', s=250)
    plt.xlabel('Probability')
    plt.yticks(ind + width/2., classes)
    plt.xlim([0, prob[imax] * 1.2])
    plt.title("%s - %s - %.2f percent" % (starttime.isoformat(), classes[imax], prob[imax]))

    plt.show()
      
def run_predict(args):

    # get configuration
    cfg = io.Config(args.config_file)

    # set up classifiers to use
    clf_dict, clf_df = clf_dict_to_sta_dataframe(cfg.clf_fname, cfg.station_names)
    with open(cfg.best_atts_fname, 'r') as f_:
        best_atts = pickle.load(f_)


    # request data from the stations
    if cfg.do_use_saved_data:
        #data_fname = os.path.join(data_dir, "55703_PF.*.MSEED")
        data_fname = os.path.join(cfg.data_dir, "39160_PF.*.MSEED")
        st = read(data_fname, starttime=args.starttime, endtime=args.starttime
                  + args.duration)
    else:
        raise NotImplementedError

    # select classifier as a function of which stations are present
    sta_names = np.unique([tr.stats.station for tr in st])
    clf_key = get_clf_key_from_stalist(clf_df, sta_names)
    clf = clf_dict[clf_key]
    best_atts_clf = best_atts[clf_key]

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

    # run prediction and output result
    X = X_df[best_atts_clf].values
    y = clf.predict(X)
    p_matrix = clf.predict_proba(X)

    # do plot and print
    print "Event is %s with probability %0.2f" % (y[0], np.max(p_matrix))
    plot_prob(p_matrix[0, :], clf.classes_, args.starttime)

if __name__ == '__main__':

    # set up parser
    cl_parser = argparse.ArgumentParser(description='Launch classification.')
    cl_parser.add_argument('config_file', help='eqdiscrim configuration file')
    cl_parser.add_argument('starttime', type=UTCDateTime, 
                           help='Timestamp of first point in window')
    cl_parser.add_argument('duration', help='Window duration in seconds', type=float)

    # parse input
    args = cl_parser.parse_args(['eqdiscrim_test.cfg', '2014-01-04T13:02:28', '7.8'])
    # args = cl_parser.parse_args()

    # run program
    run_predict(args)

