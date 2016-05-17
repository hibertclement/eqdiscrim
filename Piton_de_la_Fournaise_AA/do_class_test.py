# -*- coding: utf-8 -*-
from seis_class import evaluate_cross_validation, equalize_classes, train_and_evaluate
from sklearn.preprocessing import MinMaxScaler
import itertools
from pickle import dump, load
import numpy as np


def stat_mean(sample):
    nev = len(sample)
    mean = np.sum(sample) / nev
    return mean
    
def stat_variance( sample ) :
        n = len( sample ) # taille
        mq = stat_mean( sample )**2
        s = sum( [ x**2 for x in sample ] )
        variance = s / n - mq
        return variance
        
def stat_standard_deviation( sample ) :
        variance = stat_variance( sample )
        sd = np.sqrt( variance )
        return sd
        
def gaussian(x, mu, sig):
    return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))


    
def w_file(filename, entry_value, entry_name):
    f_ = open(filename, 'w')
    dump((entry_value, entry_name), f_)
    f_.close()
    return
    
def r_file(filename):
    f_ = open(filename, 'r')
    exit_value, exit_name = load(f_)
    f_.close()
    return exit_value, exit_name

def do_indexes(att_names, choice, n_att):
    att_indexes = []
  
    if choice == 1: 
        att_indexes.append(att_names.index('cent_f'))
        att_indexes.append(att_names.index('DUR'))
        att_indexes.append(att_names.index('K'))
        att_indexes.append(att_names.index('dur/A'))
        att_indexes.append(att_names.index('A'))
        att_indexes.append(att_names.index('dom_f'))
        att_indexes.append(att_names.index('E'))
        att_indexes.append(att_names.index('maxA_mean'))
        #att_indexes.append(att_names.index('AsDec'))
        
        comblen = [8, 28, 56, 70, 56, 28, 8, 1]
        
    elif choice == 2:
        att_indexes.append(att_names.index('cent_f')+n_att/2)
        att_indexes.append(att_names.index('DUR')+n_att/2)        
        att_indexes.append(att_names.index('K')+n_att/2)
        att_indexes.append(att_names.index('dur/A')+n_att/2)
        att_indexes.append(att_names.index('A')+n_att/2)
        att_indexes.append(att_names.index('dom_f')+n_att/2)
        att_indexes.append(att_names.index('E')+n_att/2)
        att_indexes.append(att_names.index('maxA_mean')+n_att/2)
        #att_indexes.append(att_names.index('AsDec')+n_att/2)
        
        comblen = [8, 28, 56, 70, 56, 28, 8, 1]
        
    elif choice == 'both':
        
        att_indexes.append(att_names.index('cent_f'))
        att_indexes.append(att_names.index('DUR'))
        att_indexes.append(att_names.index('K'))
        att_indexes.append(att_names.index('dur/A'))
        att_indexes.append(att_names.index('A'))
        att_indexes.append(att_names.index('dom_f'))
        att_indexes.append(att_names.index('E'))
        att_indexes.append(att_names.index('maxA_mean'))
        #att_indexes.append(att_names.index('AsDec'))
       
        att_indexes.append(att_names.index('cent_f')+n_att/2)
        att_indexes.append(att_names.index('DUR')+n_att/2)        
        att_indexes.append(att_names.index('K')+n_att/2)
        att_indexes.append(att_names.index('dur/A')+n_att/2)
        att_indexes.append(att_names.index('A')+n_att/2)
        att_indexes.append(att_names.index('dom_f')+n_att/2)
        att_indexes.append(att_names.index('E')+n_att/2)
        att_indexes.append(att_names.index('maxA_mean')+n_att/2)
        #att_indexes.append(att_names.index('AsDec')+n_att/2)
         
        comblen = []
        
    return att_indexes, comblen
        
def classification(att_indexes, X, Y, nmax, clf1, clf2, equalize_test_data=False):
    
    
    # dictionnaire qui contiennent un type de combinaisons pour tous les attributs
    Idict = {} #single
    IIdict = {} #couple
    IIIdict = {} #trio
    IVdict = {} #four
    Vdict = {} #five
    VIdict = {} #six
    VIIdict = {} #seven
    VIIIdict = {} #eight    
    
    #Idict = [ clef : [cv1, cvm1, cm1, artrain1, artest1, cv2, cvm2, cm2, artrain2, artest2] ]

    
        
    lst = att_indexes # contient tous les indexes d'attributs
    #[1, 4, 7, 6, 5, 0, 2, 8] sta1

    combs = [] # contiendra les combinaisons de x attributs
    
    LEN = [] # contient les nombres de combinaisons dans chaque dictionnaire
    
    for i in xrange(1, len(lst)+1):
        els = [list(x) for x in itertools.combinations(lst, i)]
        combs.extend(els)
        n = len(combs)
        LEN.append(n)
        
        if i == 1:
        
            for i in xrange(n):
                att_indexes = []
                att_indexes.append(combs[i][0])
                
                Xbis = X[:, att_indexes]
                
                print Idict.keys()
                print att_indexes
                
                # normalize data
                scaler = MinMaxScaler().fit(Xbis)
    
                # equalize training data
                X_train, Y_train, X_test, Y_test = equalize_classes(Xbis, Y, nmax)
                #print X_train.shape
                #print X_test.shape
    
                # equalize test data if requested
                if equalize_test_data:
                    X_tmp = X_test
                    Y_tmp = Y_test
                    X_test, Y_test, X_dump, Y_dump = equalize_classes(X_tmp, Y_tmp)
                #print X_test.shape
    


                # do cross validation
                print 'linear'
                cv1, cvm1 = evaluate_cross_validation(clf1, X_train, Y_train, 5)
                print 'rbf'
                cv2, cvm2 = evaluate_cross_validation(clf2, X_train, Y_train, 5)
            
                # create confusion matrix for test data
                print 'linear'
                cm1, artrain1, artest1 = train_and_evaluate(clf1, X_train, X_test, Y_train, Y_test)
                print 'rbf'
                cm2, artrain2, artest2 = train_and_evaluate(clf2, X_train, X_test, Y_train, Y_test)
                
                # add the value in the dictionnary
                Idict[str(combs[i])] = [cv1]
                Idict[str(combs[i])].append(cvm1)
                Idict[str(combs[i])].append(cm1)
                Idict[str(combs[i])].append(artrain1)
                Idict[str(combs[i])].append(artest1)
                
                Idict[str(combs[i])].append([cv2])
                Idict[str(combs[i])].append(cvm2)
                Idict[str(combs[i])].append(cm2)
                Idict[str(combs[i])].append(artrain2)
                Idict[str(combs[i])].append(artest2)
                
                #Idict[str(combs[i])] = [(cv1, cvm1, cm1, artrain1, artest1, cv2, cvm2, cm2, artrain2,artest2)]
                # the five first one are for the linear kernel and next for the rbf
            combs=[]
            
            
        elif i == 2 : 
            for i in xrange(n):
    
                att_indexes = []
                att_indexes.append(combs[i][0])
                att_indexes.append(combs[i][1])
                
                Xbis = X[:, att_indexes]
            
                print IIdict.keys()
                print att_indexes
                
                # normalize data
                scaler = MinMaxScaler().fit(Xbis)
    
                # equalize training data
                X_train, Y_train, X_test, Y_test = equalize_classes(Xbis, Y, nmax)
                #print X_train.shape
                #print X_test.shape
    
                # equalize test data if requested
                if equalize_test_data:
                    X_tmp = X_test
                    Y_tmp = Y_test
                    X_test, Y_test, X_dump, Y_dump = equalize_classes(X_tmp, Y_tmp)
                #print X_test.shape
    
        
                # do cross validation
                print 'linear'
                cv1, cvm1 = evaluate_cross_validation(clf1, X_train, Y_train, 5)
                print 'rbf'
                cv2, cvm2 = evaluate_cross_validation(clf2, X_train, Y_train, 5)
            
                # create confusion matrix for test data
                print 'linear'
                cm1, artrain1, artest1 = train_and_evaluate(clf1, X_train, X_test, Y_train, Y_test)
                print 'rbf'
                cm2, artrain2, artest2 = train_and_evaluate(clf2, X_train, X_test, Y_train, Y_test)
                
                # add the value in the dictionnary
                IIdict[str(combs[i])] = [cv1]
                IIdict[str(combs[i])].append(cvm1)
                IIdict[str(combs[i])].append(cm1)
                IIdict[str(combs[i])].append(artrain1)
                IIdict[str(combs[i])].append(artest1)
                
                IIdict[str(combs[i])].append([cv2])
                IIdict[str(combs[i])].append(cvm2)
                IIdict[str(combs[i])].append(cm2)
                IIdict[str(combs[i])].append(artrain2)
                IIdict[str(combs[i])].append(artest2)
                
            combs=[]

        elif i == 3 : 
            for i in xrange(len(combs)):
                
                att_indexes = []
                att_indexes.append(combs[i][0])
                att_indexes.append(combs[i][1])
                att_indexes.append(combs[i][2])
    
                Xbis = X[:, att_indexes]
                
                print IIIdict.keys()
                print att_indexes
                
                # normalize data
                scaler = MinMaxScaler().fit(Xbis)
    
                # equalize training data
                X_train, Y_train, X_test, Y_test = equalize_classes(Xbis, Y, nmax)
                #print X_train.shape
                #print X_test.shape
    
                # equalize test data if requested
                if equalize_test_data:
                    X_tmp = X_test
                    Y_tmp = Y_test
                    X_test, Y_test, X_dump, Y_dump = equalize_classes(X_tmp, Y_tmp)
                #print X_test.shape
    
               
                # do cross validation
                print 'linear'
                cv1, cvm1 = evaluate_cross_validation(clf1, X_train, Y_train, 5)
                print 'rbf'
                cv2, cvm2 = evaluate_cross_validation(clf2, X_train, Y_train, 5)
                
                # create confusion matrix for test data
                print 'linear'
                cm1, artrain1, artest1 = train_and_evaluate(clf1, X_train, X_test, Y_train, Y_test)
                print 'rbf'
                cm2, artrain2, artest2 = train_and_evaluate(clf2, X_train, X_test, Y_train, Y_test)
                
                # add the value in the dictionnary
                
                IIIdict[str(combs[i])] = [cv1]
                IIIdict[str(combs[i])].append(cvm1)
                IIIdict[str(combs[i])].append(cm1)
                IIIdict[str(combs[i])].append(artrain1)
                IIIdict[str(combs[i])].append(artest1)
       	     
                IIIdict[str(combs[i])].append([cv2])
                IIIdict[str(combs[i])].append(cvm2)
                IIIdict[str(combs[i])].append(cm2)
                IIIdict[str(combs[i])].append(artrain2)
                IIIdict[str(combs[i])].append(artest2)
                
            combs=[]
                
                
        
        elif i == 4 : 
            for i in xrange(len(combs)):
                
                att_indexes = []
                att_indexes.append(combs[i][0])
                att_indexes.append(combs[i][1])
                att_indexes.append(combs[i][2])
                att_indexes.append(combs[i][3])
                
                Xbis = X[:, att_indexes]
                
                print IVdict.keys()
                print att_indexes
                
                # normalize data
                scaler = MinMaxScaler().fit(Xbis)

                # equalize training data
                X_train, Y_train, X_test, Y_test = equalize_classes(Xbis, Y, nmax)
                #print X_train.shape
                #print X_test.shape
    
                # equalize test data if requested
                if equalize_test_data:
                    X_tmp = X_test
                    Y_tmp = Y_test
                    X_test, Y_test, X_dump, Y_dump = equalize_classes(X_tmp, Y_tmp)
                #print X_test.shape

                # do cross validation
                print 'linear'
                cv1, cvm1 = evaluate_cross_validation(clf1, X_train, Y_train, 5)
                print 'rbf'
                cv2, cvm2 = evaluate_cross_validation(clf2, X_train, Y_train, 5)
            
                # create confusion matrix for test data
                print 'linear'
                cm1, artrain1, artest1 = train_and_evaluate(clf1, X_train, X_test, Y_train, Y_test)
                print 'rbf'
                cm2, artrain2, artest2 = train_and_evaluate(clf2, X_train, X_test, Y_train, Y_test)
                
                # add the value in the dictionnary
                
                IVdict[str(combs[i])] = [cv1]
                IVdict[str(combs[i])].append(cvm1)
                IVdict[str(combs[i])].append(cm1)
                IVdict[str(combs[i])].append(artrain1)
                IVdict[str(combs[i])].append(artest1)
                
                IVdict[str(combs[i])].append([cv2])
                IVdict[str(combs[i])].append(cvm2)
                IVdict[str(combs[i])].append(cm2)
                IVdict[str(combs[i])].append(artrain2)
                IVdict[str(combs[i])].append(artest2)
                
            combs=[]
            
        elif i == 5 : 
            for i in xrange(len(combs)):
                
                att_indexes = []
                att_indexes.append(combs[i][0])
                att_indexes.append(combs[i][1])
                att_indexes.append(combs[i][2])
                att_indexes.append(combs[i][3])
                att_indexes.append(combs[i][4])
                
                Xbis = X[:, att_indexes]
                
                print Vdict.keys()
                print att_indexes
                
                # normalize data
                scaler = MinMaxScaler().fit(Xbis)
    
                # equalize training data
                X_train, Y_train, X_test, Y_test = equalize_classes(Xbis, Y, nmax)
                #print X_train.shape
                #print X_test.shape
    
                # equalize test data if requested
                if equalize_test_data:
                    X_tmp = X_test
                    Y_tmp = Y_test
                    X_test, Y_test, X_dump, Y_dump = equalize_classes(X_tmp, Y_tmp)
                #print X_test.shape
    
                # do cross validation
                print 'linear'
                cv1, cvm1 = evaluate_cross_validation(clf1, X_train, Y_train, 5)
                print 'rbf'
                cv2, cvm2 = evaluate_cross_validation(clf2, X_train, Y_train, 5)
            
                # create confusion matrix for test data
                print 'linear'
                cm1, artrain1, artest1 = train_and_evaluate(clf1, X_train, X_test, Y_train, Y_test)
                print 'rbf'
                cm2, artrain2, artest2 = train_and_evaluate(clf2, X_train, X_test, Y_train, Y_test)
                
    	        # add the value in the dictionnary
                
                Vdict[str(combs[i])] = [cv1]
                Vdict[str(combs[i])].append(cvm1)
                Vdict[str(combs[i])].append(cm1)
                Vdict[str(combs[i])].append(artrain1)
                Vdict[str(combs[i])].append(artest1)
                
                Vdict[str(combs[i])].append([cv2])
                Vdict[str(combs[i])].append(cvm2)
                Vdict[str(combs[i])].append(cm2)
                Vdict[str(combs[i])].append(artrain2)
                Vdict[str(combs[i])].append(artest2)
                
            combs=[]
        elif i == 6 : 
            for i in xrange(len(combs)):
                att_indexes = []
                att_indexes.append(combs[i][0])
                att_indexes.append(combs[i][1])
                att_indexes.append(combs[i][2])
                att_indexes.append(combs[i][3])
                att_indexes.append(combs[i][4])
                att_indexes.append(combs[i][5])
                
                Xbis = X[:, att_indexes]
                
                print VIdict.keys()
                print att_indexes
                
                # normalize data
                scaler = MinMaxScaler().fit(Xbis)
    
                # equalize training data
                X_train, Y_train, X_test, Y_test = equalize_classes(Xbis, Y, nmax)
                #print X_train.shape
                #print X_test.shape
    
                # equalize test data if requested
                if equalize_test_data:
                    X_tmp = X_test
                    Y_tmp = Y_test
                    X_test, Y_test, X_dump, Y_dump = equalize_classes(X_tmp, Y_tmp)
                #print X_test.shape

                # do cross validation
                print 'linear'
                cv1, cvm1 = evaluate_cross_validation(clf1, X_train, Y_train, 5)
                print 'rbf'
                cv2, cvm2 = evaluate_cross_validation(clf2, X_train, Y_train, 5)
            
                # create confusion matrix for test data
                print 'linear'
                cm1, artrain1, artest1 = train_and_evaluate(clf1, X_train, X_test, Y_train, Y_test)
                print 'rbf'
                cm2, artrain2, artest2 = train_and_evaluate(clf2, X_train, X_test, Y_train, Y_test)
                
                # add the value in the dictionnary
                
                VIdict[str(combs[i])] = [cv1]
                VIdict[str(combs[i])].append(cvm1)
                VIdict[str(combs[i])].append(cm1)
                VIdict[str(combs[i])].append(artrain1)
                VIdict[str(combs[i])].append(artest1)
                
                VIdict[str(combs[i])].append([cv2])
                VIdict[str(combs[i])].append(cvm2)
                VIdict[str(combs[i])].append(cm2)
                VIdict[str(combs[i])].append(artrain2)
                VIdict[str(combs[i])].append(artest2)
                    
            combs=[]
        
        elif i == 7 : 
            for i in xrange(len(combs)):
            
                att_indexes = []
                att_indexes.append(combs[i][0])
                att_indexes.append(combs[i][1])
                att_indexes.append(combs[i][2])
                att_indexes.append(combs[i][3])
                att_indexes.append(combs[i][4])
                att_indexes.append(combs[i][5])
                att_indexes.append(combs[i][6])
                
                Xbis = X[:, att_indexes]
                
                print VIIdict.keys()
                print att_indexes
                
                # normalize data
                scaler = MinMaxScaler().fit(Xbis)
    
                # equalize training data
                X_train, Y_train, X_test, Y_test = equalize_classes(Xbis, Y, nmax)
                #print X_train.shape
                #print X_test.shape
    
                # equalize test data if requested
                if equalize_test_data:
                    X_tmp = X_test
                    Y_tmp = Y_test
                    X_test, Y_test, X_dump, Y_dump = equalize_classes(X_tmp, Y_tmp)
                #print X_test.shape
    
                # do cross validation
                print 'linear'
                cv1, cvm1 = evaluate_cross_validation(clf1, X_train, Y_train, 5)
                print 'rbf'
                cv2, cvm2 = evaluate_cross_validation(clf2, X_train, Y_train, 5)
            
                # create confusion matrix for test data
                print 'linear'
                cm1, artrain1, artest1 = train_and_evaluate(clf1, X_train, X_test, Y_train, Y_test)
                print 'rbf'
                cm2, artrain2, artest2 = train_and_evaluate(clf2, X_train, X_test, Y_train, Y_test)
                
                # add the value in the dictionnary
                
                VIIdict[str(combs[i])] = [cv1]
                VIIdict[str(combs[i])].append(cvm1)
                VIIdict[str(combs[i])].append(cm1)
                VIIdict[str(combs[i])].append(artrain1)
                VIIdict[str(combs[i])].append(artest1)
            
                VIIdict[str(combs[i])].append([cv2])
                VIIdict[str(combs[i])].append(cvm2)
                VIIdict[str(combs[i])].append(cm2)
                VIIdict[str(combs[i])].append(artrain2)
                VIIdict[str(combs[i])].append(artest2)
                
            combs=[]
        
        elif i == 8 : 
            for i in xrange(len(combs)):
                
                att_indexes = []
                att_indexes.append(combs[i][0])
                att_indexes.append(combs[i][1])
                att_indexes.append(combs[i][2])
                att_indexes.append(combs[i][3])
                att_indexes.append(combs[i][4])
                att_indexes.append(combs[i][5])
                att_indexes.append(combs[i][6])
                att_indexes.append(combs[i][7])
    
                Xbis = X[:, att_indexes]
                
                print VIIIdict.keys()
                print att_indexes
                
                # normalize data
                scaler = MinMaxScaler().fit(Xbis)
    
                # equalize training data
                X_train, Y_train, X_test, Y_test = equalize_classes(Xbis, Y, nmax)
                #print X_train.shape
                #print X_test.shape
    
                # equalize test data if requested
                if equalize_test_data:
                    X_tmp = X_test
                    Y_tmp = Y_test
                    X_test, Y_test, X_dump, Y_dump = equalize_classes(X_tmp, Y_tmp)
                #print X_test.shape
    
                # do cross validation
                print 'linear'
                cv1, cvm1 = evaluate_cross_validation(clf1, X_train, Y_train, 5)
                print 'rbf'
                cv2, cvm2 = evaluate_cross_validation(clf2, X_train, Y_train, 5)
            
                # create confusion matrix for test data
                print 'linear'
                cm1, artrain1, artest1 = train_and_evaluate(clf1, X_train, X_test, Y_train, Y_test)
                print 'rbf'
                cm2, artrain2, artest2 = train_and_evaluate(clf2, X_train, X_test, Y_train, Y_test)
                
                # add the value in the dictionnary
                
                VIIIdict[str(combs[i])] = [cv1]
                VIIIdict[str(combs[i])].append(cvm1)
                VIIIdict[str(combs[i])].append(cm1)
                VIIIdict[str(combs[i])].append(artrain1)
                VIIIdict[str(combs[i])].append(artest1)
                
                VIIIdict[str(combs[i])].append([cv2])
                VIIIdict[str(combs[i])].append(cvm2)
                VIIIdict[str(combs[i])].append(cm2)
                VIIIdict[str(combs[i])].append(artrain2)
                VIIIdict[str(combs[i])].append(artest2)
                
            combs=[]
        
    print LEN
    """  COMS
         
    {'[13]': [(0.74407005520654868, 0.039313216083656954, array([[   0,    0],
        [ 565, 2105]]), 0.734375, 0.78838951310861427, 0.73438035408338087, 0.034548700626678641, array([[   0,    0],
        [ 541, 2129]]), 0.73828125, 0.79737827715355802)],
    a priori sort dans l'ordre dans lequel on les a mis 
    avec 
    Idict[str(combs[i])] = [(cv1, cvm1, cm1, artrain1, artest1, cv2, cvm2, cm2, artrain2,artest2)]
    
    
    """ 
    return Idict, IIdict, IIIdict, IVdict, Vdict, VIdict, VIIdict, VIIIdict

def orderdict(dictio):
            
    dictio_tmp = dictio.copy()
        
    #6, DurA
    dictio['1'] = [dictio_tmp.values()[1][12]]
    dictio['1'].append(dictio_tmp.values()[1][19])
    #2, E
    dictio['1'].append(dictio_tmp.values()[1][0])
    dictio['1'].append(dictio_tmp.values()[1][1])
    dictio['1'].append(dictio_tmp.values()[1][11])
    dictio['1'].append(dictio_tmp.values()[1][16])
    dictio['1'].append(dictio_tmp.values()[1][18])
    # 1, cent_f
    dictio['1'].append(dictio_tmp.values()[1][7])
    dictio['1'].append(dictio_tmp.values()[1][9])
    dictio['1'].append(dictio_tmp.values()[1][17])
    dictio['1'].append(dictio_tmp.values()[1][21])
    dictio['1'].append(dictio_tmp.values()[1][25])
    dictio['1'].append(dictio_tmp.values()[1][26])
    #5, A
    dictio['1'].append(dictio_tmp.values()[1][5])
    dictio['1'].append(dictio_tmp.values()[1][8])
    dictio['1'].append(dictio_tmp.values()[1][10])
    #0, dom_f
    dictio['1'].append(dictio_tmp.values()[1][2])
    dictio['1'].append(dictio_tmp.values()[1][3])
    dictio['1'].append(dictio_tmp.values()[1][4])
    dictio['1'].append(dictio_tmp.values()[1][13])
    dictio['1'].append(dictio_tmp.values()[1][15])
    dictio['1'].append(dictio_tmp.values()[1][22])
    dictio['1'].append(dictio_tmp.values()[1][23])
    #7, K
    dictio['1'].append(dictio_tmp.values()[1][20])
    #4, Dur
    dictio['1'].append(dictio_tmp.values()[1][6])
    dictio['1'].append(dictio_tmp.values()[1][14])
    dictio['1'].append(dictio_tmp.values()[1][24])
    dictio['1'].append(dictio_tmp.values()[1][27])
            
    return dictio
            
def extract_from_dict(dict):
    sta_cv = []
    sta_cvm = []
    sta_ar = []
    for i in xrange(len(dict)):
        sta_cv.append(dict.values()[i][0])
        sta_cvm.append(dict.values()[i][1])
        sta_ar.append(dict.values()[i][4])
    return sta_cv, sta_cvm, sta_ar