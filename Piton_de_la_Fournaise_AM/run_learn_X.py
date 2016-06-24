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

do_learning_curve = True

figdir = 'Figures'
if not os.path.exists(figdir):
    os.mkdir(figdir)

pd.set_option('mode.use_inf_as_null', True)

station_names = ["RVL", "BOR"]
max_events = 500
# station_names = ["RVL"]
att_dir = "Attributes"

#event_types = ["Effondrement", "Sommital", "Local", "Teleseisme", "Onde sonore", "Regional", "Profond"]
#event_types = ["Sommital", "Local", "Teleseisme", "Regional", "Profond", "Effondrement", "Onde sonore", "Phase T", "Indetermine"]
event_types = ["Sommital", "Local", "Teleseisme", "Regional", "Profond",
               "Effondrement", "Onde sonore", "Phase T"]
small_classes = ['Regional', 'Phase T', 'Onde sonore', 'Teleseisme', 'Local']
#large_classes = ['Effondrement', 'Sommital', 'Profond', 'Indetermine']
large_classes = ['Effondrement', 'Sommital', 'Profond']

def balance_classes(df_full, classes):
#    # only learn the classes in the header
#    print "\nExtracting small classes"
#    df_list = []
#    for ev_type in small_classes:
#        df_list.append(df_full[df_full['EVENT_TYPE'] == ev_type])
#    X_df_small = pd.concat(df_list)
#    print X_df_small['EVENT_TYPE'].value_counts()

    print "\nExtracting and resampling classes"
    df_list = []
    for ev_type in classes:
        df_list.append(df_full[df_full['EVENT_TYPE'] == ev_type].sample(n=max_events, replace=True))
    X_df = pd.concat(df_list)
    print X_df['EVENT_TYPE'].value_counts()

#    print "\nCombining 1x sampled large and 2x small classes"
#    X_df = pd.concat([X_df_small, X_df_small, X_df_large])
#    print X_df['EVENT_TYPE'].value_counts()

    return X_df

def run_classification(X_df, sta):
    # get the list of attributes we are interested in
    atts = X_df.columns[5:].values
    X = X_df[atts].values
    y = X_df['EVENT_TYPE'].values
    labels = np.unique(y)

    # use a random forest
    max_features = int(np.sqrt(len(atts)))
    clf = RandomForestClassifier(n_estimators=30, max_features=max_features)
    
    # print "Learning with a random forest"
    if do_learning_curve:
        train_sizes, train_scores, valid_scores = learning_curve(clf, X, y,
            train_sizes=[500, 1000, 1500,  2000, 2500, 2700, 2800, 2900, 3000, 3100, 3200], cv=5, scoring=OVPF_scorer)
        gr.plot_learning_curve(train_sizes, train_scores, valid_scores,
                               'Random Forest at %s' % sta, 'learn_%s.png' % sta)

    print "Producing confusion matrix"
    # Uses proportionnal splitting
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
    clf_fitted = clf.fit(X_train, y_train)
    y_pred = clf_fitted.predict(X_test)
   
    print('Classification report')
    cr = classification_report(y_test, y_pred, target_names=labels)
    print cr

    print('Cross validation scores using OVPF metric')
    cv_scores = cross_val_score(clf, X, y, scoring=OVPF_scorer, cv=5)
    score_mean = np.mean(cv_scores)
    score_2std = 2 * np.std(cv_scores)
    print("%.2f (+/-) %.2f" % (score_mean, score_2std))

    print('Confusion matrix')
    cm = confusion_matrix(y_test, y_pred)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    print cm
    # gr.plot_confusion_matrix(cm, labels, 'Random Forest at %s' % sta, 'cm_%s.png' % sta)
    gr.plot_confusion_matrix(cm_norm, labels, 'Random Forest at %s : %.2f (+/-) %.2f'
                             % (sta, score_mean, score_2std), 'cm_norm_%s.png' % sta)
 
# list for single stations
sta_X_df = []
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
    sta_X_df.append(X_df_full)

    # extract and combine classes
    X_df = balance_classes(X_df_full, event_types)

    # Run classification
    run_classification(X_df, sta)
    

# do multiple station
for i in xrange(len(station_names)):
    X_df = sta_X_df[i].copy()
    sta = station_names[i]
    atts = X_df.columns[5:].values
    new_atts = []
    for a in atts:
        new_a = '%s_%s' % (sta, a)
        X_df.rename(columns={a: new_a}, inplace=True)
        new_atts.append(new_a)
    if i == 0:
        X_multi_df = X_df.copy()
    else:
        X_multi_df = X_multi_df.join(X_df[new_atts])
    
print "\nCombined set after dropping NaNs"
X_multi_df.dropna(inplace=True)
print X_multi_df['EVENT_TYPE'].value_counts()

print "\nCombined set after class equilibration"
X_df = balance_classes(X_multi_df, event_types)
print X_multi_df['EVENT_TYPE'].value_counts()

run_classification(X_df, 'BOR+RVL')
