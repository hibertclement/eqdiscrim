import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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
      



if __name__ == '__main__':

    from obspy import UTCDateTime

    # Parameters that can be modified
    station_names = ["RVL", "BOR"]
    classes = ['Effondrement', 'Local', 'Regional', 'Sommital', 'Teleseisme', 
               'Phase T', 'Profond', 'Onde sonore']

    # Parameters that should not be modified
    clf_fname = 'clf_functions.dat'

    # set up classifiers to use
    clf_dict, clf_df = clf_dict_to_sta_dataframe(clf_fname, station_names)

    # read the argument list to get start-time and duration of event

    # request data from the stations

    # select classifier as a function of which stations are present
    clf_key = get_clf_key_from_stalist(clf_df, ["RVL"])
    clf = clf_dict[clf_key]

    # calculate attributes, combine and prune according to classifier

    # run prediction and output result

    # y = clf.predict(X)
    # p_matrix = clf.predict_proba(X)

    p_values = np.random.rand(8) 
    p_values = p_values / np.sum(p_values)
    starttime = UTCDateTime(2015, 4, 10)
    plot_prob(p_values, classes, starttime)
