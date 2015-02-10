import numpy as np
import matplotlib.pyplot as plt
from pickle import load
from cat_io.sihex_io import read_sihex_tidy
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.cross_validation import train_test_split, cross_val_score, KFold


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
Y[Y=="kr"] = "rb"
Y[Y=="sr"] = "rb"
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
X1 = X[:, :][A_cluster==1]
X2 = X[:, :][A_cluster==2]
Y0 = Y[:][A_cluster==0]
Y1 = Y[:][A_cluster==1]
Y2 = Y[:][A_cluster==2]

# count the number of each class in the 0 datasets
n_per_class = np.empty(n_class, dtype=int)
for i in xrange(n_class):
    n_per_class[i] = len(Y0[Y0==i])
n_min = np.min(n_per_class)
print "For group 0 : smallest class has %d elements"%n_min
# extract only n_min values from each class for classification
X_tmp_list = []
Y_tmp_list = []
for i in xrange(n_class):
    indexes = np.random.permutation(n_per_class[i])
    Xi = X0[:, :][Y0==i]
    Yi = Y0[:][Y0==i]
    X_tmp_list.append(Xi[indexes[0:n_min], :])
    Y_tmp_list.append(Yi[0:n_min])
X0_uni = np.vstack(X_tmp_list)
Y0_uni = np.hstack(Y_tmp_list)
print X0_uni.shape, Y0_uni.shape

# start the classification here
X_train, X_test, Y_train, Y_test = train_test_split(X0_uni, Y0_uni, test_size=0.25)

X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)


# do the scatter plot
for i in xrange(n_class):
    xs = X_train[:, 0][Y_train==i]
    ys = X_train[:, 1][Y_train==i]+np.random.randn(len(xs))*0.02
    plt.scatter(xs, ys, c=colors[i])
plt.legend(enc.classes_)
plt.xlabel('LocalHour')
plt.ylabel('LocalWeekday')
plt.show()

