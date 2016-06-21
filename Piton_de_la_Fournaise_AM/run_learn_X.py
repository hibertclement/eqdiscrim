import pickle
import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import cross_val_score
from sklearn.metrics import confusion_matrix, accuracy_score

figdir = 'Figures'
if not os.path.exists(figdir):
    os.mkdir(figdir)

pd.set_option('mode.use_inf_as_null', True)

station_names = ["RVL", "FLR", "BOR", "BON", "SNE", "FJS", "CSS", "GPS", "GPN", "FOR"]
att_dir = "Attributes"

num_train = 300

for sta in station_names:
    # read attributes
    fnames = glob.glob(os.path.join(att_dir, 'X_%s_*_dataframe.dat' % (sta)))
    if len(fnames) == 0:
        continue
    for fname in fnames:
        f_ = open(fname, 'r')
        X_df = pickle.load(f_)
        f_.close()
        if fname is fnames[0]:
            X_df_full = X_df
        else:
            X_df_full = X_df_full.append(X_df, ignore_index=False)

    X_df_full_clean = X_df_full.dropna()
    print sta
    print X_df_full_clean['EVENT_TYPE'].value_counts()

    # get the list of attributes we are interested in
    atts = X_df_full_clean.columns[5:]
    att_list = list([att for att in atts])

    # get the indexes of num_train Effondrement events
    eff_df = X_df_full_clean[X_df_full_clean['EVENT_TYPE'] == 'Effondrement']
    n = len(eff_df)
    eff_indexes =  np.random.permutation(eff_df.index.values)[0 : min(num_train, n)]

    # get the indexes of num_train Sommital events
    som_df = X_df_full_clean[X_df_full_clean['EVENT_TYPE'] == 'Sommital']
    n = len(som_df)
    som_indexes =  np.random.permutation(som_df.index.values)[0 : min(num_train, n)]

    # get the indexes of num_train Local events
    loc_df = X_df_full_clean[X_df_full_clean['EVENT_TYPE'] == 'Local']
    n = len(loc_df)
    loc_indexes =  np.random.permutation(loc_df.index.values)[0 : min(num_train, n)]

    # put them all together
    all_train_indexes = np.concatenate([eff_indexes, som_indexes, loc_indexes])
    all_test_indexes = np.concatenate([eff_df.index.values, som_df.index.values, loc_df.index.values])

    # extract the training set
    train_df = X_df_full_clean.ix[all_train_indexes]
    print train_df['EVENT_TYPE'].value_counts()
    X_train = train_df[att_list].values
    Y_train = train_df['EVENT_TYPE'].values

    test_df = X_df_full_clean.ix[all_test_indexes]
    X_test = test_df[att_list].values
    Y_test = test_df['EVENT_TYPE'].values

    # do a first classification
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
    for f in xrange(len(att_list)):
        print("Feature %d : %s (%.2f percent)" % (indices[f], att_list[indices[f]], importances[indices[f]]*100))
    
