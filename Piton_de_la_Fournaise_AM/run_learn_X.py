import pickle
import os
import glob
import string
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.cross_validation import cross_val_score, train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, precision_score, recall_score, make_scorer
from sklearn.learning_curve import learning_curve
import eqdiscrim_io as io
import eqdiscrim_graphics as gr


pd.set_option('mode.use_inf_as_null', True)

# parameters that can be modified
do_learning_curve = False
max_events = 500
n_best_atts = 10

# parameters that should not be modified
figdir = 'Figures'
att_dir = "Attributes"
best_atts_fname = 'best_attributes.dat'
clf_fname = 'clf_functions.dat'



def OVPF_score_func(y, y_pred):

    # scoring function tuned for OVPF requirements
    non_seismic_precision = precision_score(y, y_pred,
                                            labels=['Effondrement',
#                                                    'Indetermine',
                                                    'Onde sonore',
                                                    'Phase T'],
                                            average='weighted')
    seismic_recall = recall_score(y, y_pred, labels=['Sommital', 'Local',
                                                     'Regional',
                                                     'Teleseisme',
                                                     'Profond'],
                                  average='weighted')
    return np.mean((non_seismic_precision, seismic_recall))
OVPF_scorer = make_scorer(OVPF_score_func)

def balance_classes(df_full, classes):

    df_list = []
    for ev_type in classes:
        df_list.append(df_full[df_full['EVENT_TYPE'] ==
                       ev_type].sample(n=max_events, replace=True))
    X_df = pd.concat(df_list)

    return X_df

def run_classification(X_df, sta, output_info=False):

    # get the list of attributes we are interested in
    atts = X_df.columns[5:].values
    X = X_df[atts].values
    y = X_df['EVENT_TYPE'].values
    labels = np.unique(y)

    # use a random forest
    clf = RandomForestClassifier(n_estimators=100)
    
    if output_info and do_learning_curve:
        print "\nProducing learning curve"
        train_sizes, train_scores, valid_scores = learning_curve(clf, X, y,
            train_sizes=[500, 1000, 1500,  2000, 2250, 2500, 2700, 2800, 2900,
                         3000, 3100, 3200], cv=5, scoring=OVPF_scorer)
        gr.plot_learning_curve(train_sizes, train_scores, valid_scores,
                               'Random Forest at %s' % sta,
                               os.path.join(figdir, 'learn_%s.png' % sta))

    # Uses proportionnal splitting
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
    clf_fitted = clf.fit(X_train, y_train)
    y_pred = clf_fitted.predict(X_test)

  
    if output_info:
        print('\nClassification report')
        cr = classification_report(y_test, y_pred, target_names=labels)
        print cr

    if output_info:
        print('\nCross validation scores using OVPF metric')
        cv_scores = cross_val_score(clf, X, y, scoring=OVPF_scorer, cv=5)
        score_mean = np.mean(cv_scores)
        score_2std = 2 * np.std(cv_scores)
        print("%.2f (+/-) %.2f" % (score_mean, score_2std))

    if output_info:
        print('\nConfusion matrix')
        cm = confusion_matrix(y_test, y_pred)
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print cm
        gr.plot_confusion_matrix(cm_norm, labels, '%s : %.2f (+/-) %.2f'
                             % (sta, score_mean * 100, score_2std * 100),
                                os.path.join(figdir, 'cm_norm_%s.png' % sta))
 
    return clf_fitted, atts

def get_important_features(clf, atts, n_max=None):
    # get importances
    importances = clf.feature_importances_
    std = np.std([tree.feature_importances_ for tree in clf.estimators_],
             axis=0)
    indices = np.argsort(importances)[::-1]
    # indices = np.argsort(importances)
    if n_max is None or n_max > len(indices):
        return atts[indices[:]]
    else:
        return atts[indices[0:n_max]]

def combine_best_features(sta_comb, sta_X_df, sta_best_atts):

    for sta in sta_comb:
        X_df = sta_X_df[sta].copy()
        atts = X_df.columns[5:].values
        new_atts = []
        for a in atts:
            # keep attribute if in the best ones for this station
            if a in sta_best_atts[sta]:
                new_a = '%s_%s' % (sta, a)
                X_df.rename(columns={a: new_a}, inplace=True)
                new_atts.append(new_a)
            else:
                X_df.drop(a, axis=1, inplace=True)
        if sta is station_names[0]:
            X_multi_df = X_df.copy()
        else:
            X_multi_df = X_multi_df.join(X_df[new_atts])
 
    return X_multi_df


if __name__ == '__main__':

    # Parameters that can be modified
    station_names = ["RVL", "BOR"]
    event_types = ["Sommital", "Local", "Teleseisme", "Regional", "Profond",
                   "Effondrement", "Onde sonore", "Phase T"]

    # ---------------
    # CODE STARTS HERE
    # ---------------

    if not os.path.exists(figdir):
        os.mkdir(figdir)

    # ------------------------
    # single stations
    # ------------------------
    sta_X_df = {}
    sta_best_atts = {}
    sta_clf = {}
    for sta in station_names:
        # read attributes
        print "\nTreating station %s... \n" % sta
        fnames = glob.glob(os.path.join(att_dir, 'X_*_%s_*_dataframe.dat' % (sta)))
        if len(fnames) == 0:
            continue
        X_df_full = io.read_and_cat_dataframes(fnames)

        # drop all lines containing nan
        X_df_full.dropna(inplace=True)
        print X_df_full['EVENT_TYPE'].value_counts()

        # add to single station list
        sta_X_df[sta]=X_df_full

        # extract and combine classes
        X_df = balance_classes(X_df_full, event_types)

        # Run classification
        clf, atts = run_classification(X_df, sta)

        # get important features
        best_atts = get_important_features(clf, atts, n_best_atts)

        # re-run classification with best attributes only
        for a in best_atts:
            if a not in best_atts:
                X_df.drop(a, axis=1, inplace=True)
        clf, atts = run_classification(X_df, sta)

        # get important features again (in case order has changed)
        best_atts = get_important_features(clf, atts, n_best_atts)
        sta_best_atts[sta] = best_atts
        sta_clf[sta] = clf
    
    # ------------------------
    # station combinations
    # ------------------------
    station_combinations = io.get_station_combinations(station_names)

    # do for multiple stations
    for comb in station_combinations:
        sta = string.join(comb, '+')
        print "\nTreating station combination : %s...\n" % sta

        # prepare the X matrix
        X_multi_df = combine_best_features(comb, sta_X_df, sta_best_atts)
        X_multi_df.dropna(inplace=True)
        print X_df_full['EVENT_TYPE'].value_counts()

        X_df = balance_classes(X_multi_df, event_types)

        # run the combined classification
        clf, atts = run_classification(X_df, sta)
        best_atts = get_important_features(clf, atts, n_best_atts)

        # save best attributes and classifier
        sta_best_atts[sta] = best_atts
        sta_clf[sta] = clf

    # ------------------------
    # ensure permanence
    # ------------------------
    # attributes
    f_ = open(best_atts_fname, 'w')
    pickle.dump(sta_best_atts, f_)
    f_.close()

    # classifiers
    f_ = open(clf_fname, 'w')
    pickle.dump(sta_clf, f_)
    f_.close()

