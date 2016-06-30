import pickle
import os
import glob
import string
import argparse
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import cross_val_score, train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import precision_score, recall_score, make_scorer
from sklearn.learning_curve import learning_curve
import eqdiscrim_io as io
import eqdiscrim_graphics as gr


pd.set_option('mode.use_inf_as_null', True)


def OVPF_score_func(y, y_pred):

    # scoring function tuned for OVPF requirements
    non_seismic_precision = precision_score(y, y_pred,
                                            labels=['Effondrement',
                                                    'Onde sonore',
                                                    'Phase T'],
                                            average='weighted')
    seismic_recall = recall_score(y, y_pred, labels=['Sommital',
                                                     'Local',
                                                     'Regional',
                                                     'Teleseisme',
                                                     'Profond'],
                                  average='weighted')
    return np.mean((non_seismic_precision, seismic_recall))

OVPF_scorer = make_scorer(OVPF_score_func)


def balance_classes(cfg, df_full, classes):

    df_list = []
    for ev_type in classes:
        try:
            df_list.append(df_full[df_full['EVENT_TYPE'] ==
                                   ev_type].sample(n=cfg.max_events,
                                                   replace=True))
        except ValueError:
            continue
    X_df = pd.concat(df_list)

    return X_df


def run_classification(cfg, X_df, sta, cross_valid=False, output_info=False):

    # get the list of attributes we are interested in
    atts = X_df.columns[5:].values
    atts_with_analyst = X_df.columns[4:].values
    X = X_df[atts_with_analyst].values
    y = X_df['EVENT_TYPE'].values
    labels = np.unique(y)

    # use a random forest
    clf = RandomForestClassifier(n_estimators=100, class_weight='balanced')

    if output_info and cfg.do_learning_curve:
        print "\nProducing learning curve"
        train_sizes, train_scores, valid_scores =\
            learning_curve(clf, X[:, 1:], y,
                           train_sizes=[500, 1000, 1500,  2000, 2250, 2500,
                                        2700, 2800, 2900, 3000, 3100, 3200],
                           cv=5, scoring=OVPF_scorer)
        gr.plot_learning_curve(train_sizes, train_scores, valid_scores,
                               'Random Forest at %s' % sta,
                               os.path.join(cfg.figdir, 'learn_%s.png' % sta))

    # Uses proportionnal splitting
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
    
    # set weights according to analyst quality
    sample_weight = np.ones(len(y_train)) * 0.5
    for analyst in cfg.good_analysts:
        sample_weight[X_train[:, 0] == analyst] = 1.0

    # fit classifier on training data
    clf_fitted = clf.fit(X_train[:, 1:], y_train, sample_weight=sample_weight)

    # predict on test data
    y_pred = clf_fitted.predict(X_test[:, 1:])

    # get cross_validation score
    if cross_valid:
        print('\nCross validation scores using OVPF metric')
        cv_scores = cross_val_score(clf, X[:, 1:], y, scoring=OVPF_scorer, cv=5)
        score_mean = np.mean(cv_scores)
        score_2std = 2 * np.std(cv_scores)
        print("%.2f (+/-) %.2f" % (score_mean, score_2std))
    else:
        score_mean = None

    if output_info:
        print('\nClassification report')
        cr = classification_report(y_test, y_pred, target_names=labels)
        print cr


    if output_info:
        print('\nConfusion matrix')
        cm = confusion_matrix(y_test, y_pred)
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print cm
        gr.plot_confusion_matrix(cm_norm, labels, '%s : %.2f (+/-) %.2f'
                                 % (sta, score_mean * 100, score_2std * 100),
                                 os.path.join(cfg.figdir,
                                              'cm_norm_%s.png' % sta))

    return clf_fitted, atts, score_mean


def get_important_features(clf, atts, n_max=None):
    # get importances
    importances = clf.feature_importances_
    indices = np.argsort(importances)[::-1]
    if n_max is None or n_max > len(indices):
        return atts[indices[:]]
    else:
        return atts[indices[0:n_max]]


def combine_best_features(sta_comb, sta_X_df, sta_best_atts, station_names):

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
        try:
            X_multi_df = X_multi_df.join(X_df[new_atts])
        except UnboundLocalError:
            X_multi_df = X_df.copy()

    return X_multi_df


def run_learn(args):

    cfg = io.Config(args.config_file)

    if not os.path.exists(cfg.figdir):
        os.mkdir(cfg.figdir)

    if not os.path.exists(cfg.clfdir):
        os.mkdir(cfg.clfdir)

    # ------------------------
    # single stations
    # ------------------------
    sta_X_df = {}
    sta_best_atts = {}
    with open(cfg.scores_fname, 'w') as f_:
        f_.write("STA_COMB, SCORE\n")
    for sta in cfg.station_names:
        # read attributes
        print "\nTreating station %s... \n" % sta
        fnames = glob.glob(os.path.join(cfg.att_dir,
                                        'X_*_%s_*_dataframe.dat' % (sta)))
        if len(fnames) == 0:
            continue
        X_df_full = io.read_and_cat_dataframes(fnames)

        # drop all lines containing nan
        X_df_full.dropna(inplace=True)
        print X_df_full['EVENT_TYPE'].value_counts()

        # add to single station list
        sta_X_df[sta] = X_df_full

        # extract and combine classes
        X_df = balance_classes(cfg, X_df_full, cfg.event_types)
        # X_df = X_df_full.copy()

        # Run classification
        clf, atts, score = run_classification(cfg, X_df, sta)

        # get important features
        best_atts = get_important_features(clf, atts, cfg.n_best_atts)

        # re-run classification with best attributes only
        for a in atts:
            if a not in best_atts:
                X_df.drop(a, axis=1, inplace=True)
        clf, atts, score = run_classification(cfg, X_df, sta, cross_valid=True,
                                              output_info=cfg.output_info)

        clf_fname = os.path.join(cfg.clfdir, 'clf_%s.dat' % sta)
        with open(clf_fname, 'w') as f_:
            pickle.dump(clf, f_)
        with open(cfg.scores_fname, 'a') as f_:
            f_.write("%s %.2f\n" % (sta, score))
        sta_best_atts[sta] = X_df.columns.values[5:]

    # ------------------------
    # station combinations
    # ------------------------
    station_combinations = io.get_station_combinations(cfg.station_names,
                                                      cfg.n_stations_per_group)

    # do for multiple stations
    for comb in station_combinations:
        sta = string.join(comb, '+')
        print "\nTreating station combination : %s...\n" % sta

        # prepare the X matrix
        X_multi_df = combine_best_features(comb, sta_X_df, sta_best_atts,
                                           cfg.station_names)
        X_multi_df.dropna(inplace=True)
        print X_df_full['EVENT_TYPE'].value_counts()

        X_df = balance_classes(cfg, X_multi_df, cfg.event_types)
        # X_df = X_df_full.copy()

        # run the combined classification
        clf, atts, score = run_classification(cfg, X_df, sta)
        best_atts = get_important_features(clf, atts, cfg.n_best_atts)

        # re-run classification with best attributes only
        for a in atts:
            if a not in best_atts:
                X_df.drop(a, axis=1, inplace=True)
        clf, atts, score = run_classification(cfg, X_df, sta, cross_valid=True,
                                              output_info=cfg.output_info)

        # save best attributes and classifier and score
        sta_best_atts[sta] = X_df.columns.values[5:]
        clf_fname = os.path.join(cfg.clfdir, 'clf_%s.dat' % sta)
        with open(clf_fname, 'w') as f_:
            pickle.dump(clf, f_)
        with open(cfg.scores_fname, 'a') as f_:
            f_.write("%s %.2f\n" % (sta, score))

    # ------------------------
    # ensure permanence
    # ------------------------
    # attributes
    with open(cfg.best_atts_fname, 'w') as f_:
        pickle.dump(sta_best_atts, f_)

    # scores


if __name__ == '__main__':

    # set up parser
    parser = \
        argparse.ArgumentParser(
            description='Launch attribute calculation for classifier training')
    parser.add_argument('config_file', help='eqdiscrim configuration file')

    # parse input
    args = parser.parse_args()

    # run program
    run_learn(args)
