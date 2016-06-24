import pickle
import os
import glob
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


# ----------------
# make OVPF scorer
def OVPF_score_func(y, y_pred):
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
# ----------------

do_learning_curve = False

figdir = 'Figures'
if not os.path.exists(figdir):
    os.mkdir(figdir)

pd.set_option('mode.use_inf_as_null', True)

station_names = ["RVL", "BOR"]
max_events = 500
n_best_atts = 10
att_dir = "Attributes"
best_atts_fname = 'best_attributes.dat'

event_types = ["Sommital", "Local", "Teleseisme", "Regional", "Profond",
               "Effondrement", "Onde sonore", "Phase T"]

def balance_classes(df_full, classes):

    print "\nExtracting and resampling classes"
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
    max_features = int(np.sqrt(len(atts)))
    clf = RandomForestClassifier(n_estimators=100, max_features=max_features)
    
    if output_info and do_learning_curve:
        print "Learning with a random forest"
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
        print('Classification report')
        cr = classification_report(y_test, y_pred, target_names=labels)
        print cr

    if output_info:
        print('Cross validation scores using OVPF metric')
        cv_scores = cross_val_score(clf, X, y, scoring=OVPF_scorer, cv=5)
        score_mean = np.mean(cv_scores)
        score_2std = 2 * np.std(cv_scores)
        print("%.2f (+/-) %.2f" % (score_mean, score_2std))

    if output_info:
        print('Confusion matrix')
        cm = confusion_matrix(y_test, y_pred)
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print cm
        gr.plot_confusion_matrix(cm_norm, labels, '%s : %.2f (+/-) %.2f'
                             % (sta, score_mean * 100, score_2std * 100),
                                os.path.join(figdir, 'cm_norm_%s.png' % sta))
 
    return clf_fitted, atts

def get_important_features(clf, atts, n_max=None):
    # get importances
    print "Most important features"
    importances = clf.feature_importances_
    std = np.std([tree.feature_importances_ for tree in clf.estimators_],
             axis=0)
    indices = np.argsort(importances)[::-1]
    # indices = np.argsort(importances)
    if n_max is None or n_max > len(indices):
        return atts[indices[:]]
    else:
        return atts[indices[0:n_max]]

# ---------------
# CODE START HERE
# ---------------

# list for single stations
sta_X_df = {}
sta_best_atts = {}
for sta in station_names:
    # read attributes
    print "Treating station %s" % sta
    fnames = glob.glob(os.path.join(att_dir, 'X_*_%s_*_dataframe.dat' % (sta)))
    if len(fnames) == 0:
        continue
    print "Reading and concatenating %d dataframes" % len(fnames)
    X_df_full = io.read_and_cat_dataframes(fnames)

    # drop all lines containing nan
    print "Dropping all rows containing NaN"
    X_df_full.dropna(inplace=True)
    print X_df_full['EVENT_TYPE'].value_counts()

    # add to single station list
    sta_X_df[sta]=X_df_full

    # extract and combine classes
    X_df = balance_classes(X_df_full, event_types)

    # Run classification
    clf, atts = run_classification(X_df, sta, output_info=False)

    # get important features
    best_atts = get_important_features(clf, atts, n_best_atts)
    sta_best_atts[sta] = best_atts

    # re-run classification with best attributes only
    for a in best_atts:
        if a not in best_atts:
            X_df.drop(a, axis=1, inplace=True)
    clf, atts = run_classification(X_df, sta, output_info=True)
    
# save best attributes dictionnary for permanence
f_ = open(best_atts_fname, 'w')
pickle.dump(sta_best_atts, f_)
f_.close()

# do multiple station
for sta in station_names:
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
    
print "\nCombined set after dropping NaNs"
X_multi_df.dropna(inplace=True)
print X_multi_df['EVENT_TYPE'].value_counts()

print "\nCombined set after class equilibration"
X_df = balance_classes(X_multi_df, event_types)
print X_df['EVENT_TYPE'].value_counts()

clf, best_atts = run_classification(X_df, 'BOR+RVL', output_info=True)
