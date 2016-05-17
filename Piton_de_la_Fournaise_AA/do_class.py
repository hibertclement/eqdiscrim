# -*- coding: utf-8 -*-
from seis_class import clean_datasets
from do_class_test import do_indexes
from do_class_test import classification
from do_class_test import w_file, r_file, stat_mean, orderdict
from pickle import dump
from sklearn.svm import SVC

# 2250 som, 727 eff in all the catalogue
# 2926 som, 256 eff for vf
# if you don't remember if you are in vf or all cat, do explore_data  and look 
# at the histogrammes !

nsta = 8 # 4 LB, 4 CP     

X_FILENAME = ['XRF.dat', 'XGS.dat', 'XBD.dat', 'XEP.dat']
Y_FILENAME = ['YRF.dat', 'YGS.dat', 'YBD.dat', 'YEP.dat']

# clusterizer
clf1 = SVC(kernel='linear')
clf2 = SVC(kernel='rbf')

# dump the clf
f_ = open('clf1.dat', 'w')
dump(clf1, f_)
f_.close()

f_ = open('clf2.dat', 'w')
dump(clf2, f_)
f_.close()


# choose the couple of stations
for e in xrange(nsta/2):
    X_filename = X_FILENAME[e]
    Y_filename = Y_FILENAME[e]
    
    nmax = 500 # si change ici, change plus bas pour les plot 2 à 2
    equalize_test_data = True
    
    # read data and take only the chosen events 
    X, att_names, Y, Ydict = clean_datasets(X_filename, Y_filename)
    n_ev, n_att = X.shape
    
    # choose the station 
    # 2 for the sta2, 1 for the sta1, 'both' for the two stations
    
    for f in xrange(2):
        choice = f+1
    
        # take the index of the attributs
        # give the number of combinations for one att, for two att, ...etc
        
        att_indexes, comblen = do_indexes(att_names, choice, n_att)
            
        # make the dict in which the value of cv, cvm, ar are stored
        nit = 10 # repeat nit times 
        n = n_att/2 -1
        
        for i in xrange(nit):
            
            Idict, IIdict, IIIdict, IVdict, Vdict, VIdict, VIIdict, VIIIdict = classification(att_indexes, X, Y, nmax, clf1, clf2, equalize_test_data=True)
            print VIIIdict
            print Idict
        
            for j in xrange(n):
            # store the dict in a file
            #1dict0 first dict with the result for each combi of 1 att
            #1dict1 second ...
            
                filename = '%d%d%ddict%d.dat'%(e,choice,j,i)
            
                if j == 0:
                    w_file(filename, Idict.values(), Idict.keys())
                if j == 1:
                    w_file(filename, IIdict.values(), IIdict.keys())
                if j == 2:
                    w_file(filename, IIIdict.values(), IIIdict.keys()) 
                if j == 3:
                    w_file(filename, IVdict.values(), IVdict.keys()) 
                if j == 4:
                    w_file(filename, Vdict.values(), Vdict.keys()) 
                if j == 5:
                    w_file(filename, VIdict.values(), VIdict.keys()) 
                if j == 6:
                    w_file(filename, VIIdict.values(), VIIdict.keys()) 
                if j == 7:
                    w_file(filename, VIIIdict.values(), VIIIdict.keys())
                    
        # dict
        #CVdict
        #{0: [0.4375, 0.66874999999999996, 0.58437500000000009, 0.66874999999999996, 0.47812500000000002, 0.45937499999999998, 
        #0.54374999999999996, 0.71250000000000002], 1: [0.70625000000000004, 0.68124999999999991, 0.71250000000000002, 0.640625, 
        #0.62812500000000004, 0.71875, 0.609375, 0.62187499999999996, 0.51875000000000004, 0.56874999999999998, 0.50312500000000004, 
        #0.734375, 0.640625, 0.74062499999999998, 0.50937500000000002, 0.453125, 0.76249999999999996, 0.74062499999999998, 
        #0.52187500000000009, 0.44687500000000002, 0.72812500000000002, 0.70937500000000009, 0.72500000000000009, 0.55625000000000002, 
        #0.71250000000000002, 0.69687500000000002, 0.64687500000000009, 0.73750000000000004], ...}
        # { 0 : [], ..
        # the values are all of the cv result for each combination of x att
        # the keys are the type of combinations
        # 0 means combinations of 1 att
        # 1 means combination of 2 att, etc...
        
    
        CVdict = {}
        CVMdict = {}
        ARdict = {}
        
        n = n_att/2 -1 # we use only 8/9 attributs, n_att = 18
        
        for k in xrange(n):   # as much dict as d'attributs,  because for 8 att, 8 type of possible combi 
            natt = comblen[k]
            
            M_cv = [] # mean of all cv result for one dict
            M_cvm = []
            M_ar = []
            
            for i in xrange(natt):
                
                m_cv = [] 
                m_cvm = []
                m_ar = []
        
                for j in xrange(nit):
                    
                    filename = '%d%d%ddict%d.dat'%(e,choice,k,j)
        
                    val, key = r_file(filename)
                    
                    m_cv.append(val[i][0])
                    m_cvm.append(val[i][1])
                    m_ar.append(val[i][4]) # ar for test set
                    
                M_cv.append(stat_mean(m_cv))
                M_cvm.append(stat_mean(m_cvm))
                M_ar.append(stat_mean(m_ar))
    
            
            CVdict[k] = M_cv # for the i-type of combi, all the mean for each combi
            CVMdict[k] = M_cvm
            ARdict[k] = M_ar
        
        # order dictionary
        
        CVdict = orderdict(CVdict)     
        CVMdict = orderdict(CVMdict)
        ARdict = orderdict(ARdict)
        
    
            
        filenameI = 'CVdict%d%d.dat'%(e,choice)
        filenameII = 'CVMdict%d%d.dat'%(e,choice)
        filenameIII = 'ARdict%d%d.dat'%(e,choice)
        
        w_file(filenameI, CVdict.values(), CVdict.keys())
        w_file(filenameII, CVMdict.values(), CVMdict.keys())
        w_file(filenameIII, ARdict.values(), ARdict.keys())
        
        
        
        
        
        
    """  COMS
        
    {'[13]': [(0.74407005520654868, 0.039313216083656954, array([[   0,    0],
        [ 565, 2105]]), 0.734375, 0.78838951310861427, 0.73438035408338087, 0.034548700626678641, array([[   0,    0],
        [ 541, 2129]]), 0.73828125, 0.79737827715355802)],
    a priori sort dans l'ordre dans lequel on les a mis 
    avec 
    Idict[str(combs[i])] = [(cv1, cvm1, cm1, artrain1, artest1, cv2, cvm2, cm2, artrain2,artest2)]
    
    
    """
    

