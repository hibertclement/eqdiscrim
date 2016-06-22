import pickle
import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import cross_val_score
from sklearn.metrics import confusion_matrix, accuracy_score
import eqdiscrim_io as io

figdir = 'Figures'
if not os.path.exists(figdir):
    os.mkdir(figdir)

pd.set_option('mode.use_inf_as_null', True)

station_names = ["RVL", "FLR", "BOR", "BON", "SNE", "FJS", "CSS", "GPS", "GPN", "FOR"]
att_dir = "Attributes"

event_types = ["Effondrement", "Sommital", "Local", "Teleseisme", "Onde sonore"]
num_train = 500

for sta in station_names:
    # read attributes
    print "Treating station %s" % sta
    fnames = glob.glob(os.path.join(att_dir, 'X_%s_*_dataframe.dat' % (sta)))
    if len(fnames) == 0:
        continue
    print "Reading and concatenating %d dataframes" % len(fnames)
    X_df_full = io.read_and_cat_dataframes(fnames)

    # drop all lines containing nan
    print "Dropping all rows containing NaN"
    X_df_full.dropna(inplace=True)
    print X_df_full['EVENT_TYPE'].value_counts()

    # get the list of attributes we are interested in
    atts = X_df_full.columns[5:]
    att_list = list([att for att in atts])

    # extract the subsets according to type
    print "Extracting events according to type"
    df_list = []
    df_indexes_list = []
    for evtype in event_types:
        df = X_df_full[X_df_full['EVENT_TYPE'] == evtype]
        n = len(df)
        df_indexes = np.random.permutation(df.index.values)[0 : min(num_train, n)]
        df_list.append(df)
        df_indexes_list.append(df_indexes)

    # put them all together
    all_train_indexes = np.concatenate(df_indexes_list)
    all_test_indexes = np.concatenate([df.index.values for df in df_list])

    # extract the training set
    train_df = X_df_full.ix[all_train_indexes]
    print train_df['EVENT_TYPE'].value_counts()
    X_train = train_df[att_list].values
    Y_train = train_df['EVENT_TYPE'].values

    test_df = X_df_full.ix[all_test_indexes]
    X_test = test_df[att_list].values
    Y_test = test_df['EVENT_TYPE'].values

    # do a first classification
    max_features = int(np.sqrt(len(att_list)))
    clf = RandomForestClassifier(n_estimators=30, max_features=max_features)
    clf = clf.fit(X_train, Y_train)
    # score on training set
    Y_pred = clf.predict(X_train)
    scores = cross_val_score(clf, X_train, Y_train, cv=5)
    cm = confusion_matrix(Y_train, Y_pred)
    print "Training accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2)
    print cm
    # score on test set
    Y_pred = clf.predict(X_test)
    cm = confusion_matrix(Y_test, Y_pred)
    print "Testing accuracy: %0.2f" % (accuracy_score(Y_test, Y_pred))
    print cm

    # print importances
    importances = clf.feature_importances_
    std = np.std([tree.feature_importances_ for tree in clf.estimators_], axis=0)
    indices = np.argsort(importances)[::-1]

    # print("Feature ranking:")
    # for f in xrange(len(att_list)):
    #    print("Feature %d : %s (%.2f percent)" % (indices[f], att_list[indices[f]], importances[indices[f]]*100))
    
    # now do it again but with only best attributes
    print "Using 10 best features"
    best_atts = [att_list[indices[i]] for i in xrange(10)]
    X_train = train_df[best_atts].values
    X_test = test_df[best_atts].values

    clf = RandomForestClassifier(n_estimators = 10)
    clf = clf.fit(X_train, Y_train)
    # score on training set
    Y_pred = clf.predict(X_train)
    scores = cross_val_score(clf, X_train, Y_train, cv=5)
    cm = confusion_matrix(Y_train, Y_pred)
    print "Training accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2)
    print cm
    # score on test set
    Y_pred = clf.predict(X_test)
    cm = confusion_matrix(Y_test, Y_pred)
    print "Testing accuracy: %0.2f" % (accuracy_score(Y_test, Y_pred))
    print cm

    # print importances
    importances = clf.feature_importances_
    std = np.std([tree.feature_importances_ for tree in clf.estimators_], axis=0)
    indices = np.argsort(importances)[::-1]

    print("Feature ranking:")
    for f in xrange(len(best_atts)):
        print("Feature %d : %s (%.2f percent)" % (indices[f], att_list[indices[f]], importances[indices[f]]*100))



