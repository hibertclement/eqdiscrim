import numpy as np
import matplotlib.pyplot as plt
from pickle import load
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.cross_validation import train_test_split, cross_val_score, KFold
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn import metrics
from cat_io.sihex_io import read_sihex_tidy
from graphics.graphics_class import plot_2D_scatter, plot_confusion_matrix
from preproc import equalize_classes


Atable = '../static_catalogs/sihex_tidy_all.dat'
Hfile = '../static_catalogs/sihex_tidy_header.txt'

clust_file = 'clusteriser.dat'

colors = ['red', 'blue', 'green', 'cyan']

# read the sihex table
A, coldict = read_sihex_tidy(Atable, Hfile)
print coldict

# read the clusterizer (based on 3-station distance)
f_ = open(clust_file, 'r')
clf = load(f_)
f_.close()

# clump the classes together as suggested by exploratory tests
# an = km + sm + me (anthropogenic events)
# ud = ki + si + uk (uniformly distributed events)
# rb = kr + sr (rock-bursts - strange time distribution)
Y = A[:, coldict['Type']]
Y[Y=="km"] = "an"
Y[Y=="sm"] = "an"
Y[Y=="me"] = "an"
Y[Y=="ki"] = "ud"
Y[Y=="si"] = "ud"
Y[Y=="uk"] = "ud"
Y[Y=="ke"] = "ud"
Y[Y=="kr"] = "ud"
Y[Y=="sr"] = "ud"
# turn them into integer labels
enc = LabelEncoder().fit(Y)
n_class = len(enc.classes_)
print enc.classes_, enc.transform(enc.classes_)
Y = enc.transform(Y)

# get clusters indexes
i1 = coldict['DistanceStation1']
i2 = coldict['DistanceStation2']
i3 = coldict['DistanceStation3']
A_cluster = clf.predict(A[:, [i1, i2, i3]])

# extract the features we want to classify on
X = A[:, (coldict['LocalHour'], coldict['LocalWeekday'])]
X *= 1.0

# do scaling on whole dataset
scaler = MinMaxScaler().fit(X)

# separate out into the three station-distance clusters
X0 = X[:, :][A_cluster==0]
Y0 = Y[:][A_cluster==0]

# count the number of each class in the 0 datasets
X0_uni, Y0_uni = equalize_classes(X0, Y0)
X_uni, Y_uni = equalize_classes(X, Y)

print X0_uni.shape, Y0_uni.shape
print X_uni.shape, Y_uni.shape

# start the classification here
X0_train, X0_test, Y0_train, Y0_test = train_test_split(X0_uni, Y0_uni,
                                                        test_size=0.25)
X_train, X_test, Y_train, Y_test = train_test_split(X_uni, Y_uni,
                                                        test_size=0.25)


plot_2D_scatter(X0_train, X0_test, Y0_train, Y0_test, enc.classes_,
                'LocalHour', 'LocalWeekday',
                'preclass0_LocalHour_LocalWeekday.png', yjitter=0.15)
plot_2D_scatter(X_train, X_test, Y_train, Y_test, enc.classes_,
                'LocalHour', 'LocalWeekday',
                'preclass_LocalHour_LocalWeekday.png', yjitter=0.15)


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
    #print "Classification report:"
    #print metrics.classification_report(Y_test, Y_pred)

    print "Confusion matrix:"
    cm = metrics.confusion_matrix(Y_test, Y_pred)
    print cm
    return cm

# do the classification

clf = Pipeline([('scaler', scaler),
                ('clf', SVC(kernel='rbf'))])

print 'Group 0'
#evaluate_cross_validation(clf, X0_train, Y0_train, 5)
cm = train_and_evaluate(clf, X0_train, X0_test, Y0_train, Y0_test)
plot_confusion_matrix(cm, enc.classes_, 'Group 0', 'cm_group0.png')
#print 'All'
#evaluate_cross_validation(clf, X_train, Y_train, 5)
#cm = train_and_evaluate(clf, X_train, X_test, Y_train, Y_test)

