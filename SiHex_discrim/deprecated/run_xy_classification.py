import numpy as np
from pickle import load
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.cross_validation import train_test_split, cross_val_score, KFold
from sklearn.metrics import confusion_matrix
from cat_io.sihex_io import read_sihex_tidy
from preproc import equalize_classes
from graphics.graphics_class import plot_confusion_matrix

Atable = '../static_catalogs/sihex_tidy_all.dat'
Hfile = '../static_catalogs/sihex_tidy_header.txt'

clust_file = 'clusteriser.dat'

clf = Pipeline([('scaler', MinMaxScaler()),
                ('clf', KNeighborsClassifier(n_neighbors=5,
                                             weights='distance'))])

def evaluate_cross_validation(clf, X, Y, k):
    # create a k-fold cross-validation iterator of k-folds
    cv = KFold(len(Y), k, shuffle=True)
    scores = cross_val_score(clf, X, Y, cv=cv)
    print "Mean score: %.3f +/- %.3f"%(np.mean(scores), np.std(scores))

def train_and_evaluate(clf, X_train, X_test, Y_train, Y_test):
    clf.fit(X_train, Y_train)
    print "Accuracy on training set: %.2f"%clf.score(X_train, Y_train)
    print "Accuracy on testing set: %.2f"%clf.score(X_test, Y_test)

    Y_pred = clf.predict(X_test)

    print "Confusion matrix:"
    cm = confusion_matrix(Y_test, Y_pred)
    print cm
    return cm

# read the sihex table
A, coldict = read_sihex_tidy(Atable, Hfile)
print coldict

# read the clusterizer (based on 3-station distance)
f_ = open(clust_file, 'r')
clust = load(f_)
f_.close()

# clump the classes together as suggested by exploratory tests
# an = km + sm + me (anthropogenic events)
# ud = ki + si + uk (uniformly distributed events)
# rb = kr + sr (rock-bursts - strange time distribution)
Y = A[:, coldict['Type']]
Y[Y=="km"] = "ksm"
Y[Y=="sm"] = "ksm"
Y[Y=="me"] = "ksm"
Y[Y=="ki"] = "ksi"
Y[Y=="si"] = "ksi"
Y[Y=="kr"] = "ksr"
Y[Y=="sr"] = "ksr"
Y[Y=="uk"] = "kse"
Y[Y=="ke"] = "kse"
Y[Y=="se"] = "kse"
# turn them into integer labels
enc = LabelEncoder().fit(Y)
n_class = len(enc.classes_)
print enc.classes_, enc.transform(enc.classes_)
Y = enc.transform(Y)

# get clusters indexes
i1 = coldict['DistanceStation1']
i2 = coldict['DistanceStation2']
i3 = coldict['DistanceStation3']
A_cluster = clust.predict(A[:, [i1, i2, i3]])
print A_cluster.shape

# extract the features we want to classify on
X = A[:, (coldict['X'], coldict['Y'], coldict['LocalHour'],
                   coldict['LocalWeekday'])]
X *= 1.0

# separate out closest distance cluster
X_0 = X[:, :][A_cluster==0]
Y_0 = Y[:][A_cluster==0]

# equalize classes
X_uni, Y_uni = equalize_classes(X, Y)
X_0_uni, Y_0_uni = equalize_classes(X_0, Y_0)
print X_uni.shape, Y_uni.shape, X_0_uni.shape, Y_0_uni.shape

title = "All points, space only"
print title
X_train, X_test, Y_train, Y_test = train_test_split(X_uni[:, 0:2], Y_uni,
                                                    test_size=0.25)
evaluate_cross_validation(clf, X_train, Y_train, 5)
cm = train_and_evaluate(clf, X_train, X_test, Y_train, Y_test)
plot_confusion_matrix(cm, enc.classes_, 'All points, space',
                      'cm_allpoints_space_classification.png')

title = "All points, space and time"
print title
X_train, X_test, Y_train, Y_test = train_test_split(X_uni[:, :], Y_uni,
                                                    test_size=0.25)
evaluate_cross_validation(clf, X_train, Y_train, 5)
cm = train_and_evaluate(clf, X_train, X_test, Y_train, Y_test)
plot_confusion_matrix(cm, enc.classes_, 'All points, space+time',
                      'cm_allpoints_spacetime_classification.png')
