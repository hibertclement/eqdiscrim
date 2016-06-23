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
                                                    'Indetermine',
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

figdir = 'Figures'
if not os.path.exists(figdir):
    os.mkdir(figdir)

pd.set_option('mode.use_inf_as_null', True)

#station_names = ["RVL", "BOR"]
station_names = ["RVL"]
att_dir = "Attributes_old"

#event_types = ["Effondrement", "Sommital", "Local", "Teleseisme", "Onde sonore", "Regional", "Profond"]
event_types = ["Sommital", "Local", "Teleseisme", "Regional", "Profond", "Effondrement", "Onde sonore", "Phase T", "Indetermine"]

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

    # only learn the classes in the header
    print "Extracting chosen classes"
    df_list = []
    for ev_type in event_types:
        df_list.append(X_df_full[X_df_full['EVENT_TYPE'] == ev_type])
    X_df = pd.concat(df_list)
    print X_df['EVENT_TYPE'].value_counts()

    # get the list of attributes we are interested in
    atts = X_df.columns[5:].values
    X = X_df[atts].values
    y = X_df['EVENT_TYPE'].values

    # use a random forest
    max_features = int(np.sqrt(len(atts)))
    clf = RandomForestClassifier(n_estimators=30, max_features=max_features)
    
    # print "Learning with a random forest"
    train_sizes, train_scores, valid_scores = learning_curve(clf, X, y,
        train_sizes=[300, 600, 1000, 1300, 1600,  2000, 2100, 2200], cv=5, scoring=OVPF_scorer)
#        train_sizes=[300, 600, 900, 1000, 1200, 1400, 1600, 1800, 1900], cv=5, scoring=OVPF_scorer)
    gr.plot_learning_curve(train_sizes, train_scores, valid_scores, 'Random Forest at %s' % sta, 'learn_%s.png' % sta)

    print "Producing confusion matrix"
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
    clf_fitted = clf.fit(X_train, y_train)
    y_pred = clf_fitted.predict(X_test)

    print('Confusion matrix')
    cm = confusion_matrix(y_test, y_pred)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    labels = np.unique(y)
    print cm
    # gr.plot_confusion_matrix(cm, labels, 'Random Forest at %s' % sta, 'cm_%s.png' % sta)
    gr.plot_confusion_matrix(cm_norm, labels, 'Random Forest at %s' % sta, 'cm_norm_%s.png' % sta)
    
    print('Classification report')
    cr = classification_report(y_test, y_pred, target_names=labels)
    print cr

    print('Cross validation scores using OVPF metric')
    cv_scores = cross_val_score(clf, X, y, scoring=OVPF_scorer, cv=5)
    print("%.2f (+/-) %.2f" % (np.mean(cv_scores), np.std(cv_scores)))
