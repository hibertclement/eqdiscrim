# -*- coding: utf-8 -*-
import numpy as np 
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

""" ON PROGRESS
def plot_class(X_eff, X_som, sta, att_names,att_indexes):

    sta == 1 for the station 1
    sta == 2 for the station 2
    
    x= att_names[att_indexes[0]]
    y= att_names[att_indexes[1]]
    if sta==1:
    #station 1
        X_eff_sta1 = X_eff[:,0:n_attind]  
        X_som_sta1 = X_som[:,0:n_attind]
    
        plt.scatter(X_eff_sta1[:,0],X_eff_sta1[:,1], c='r', marker='x') 
        plt.scatter(X_som_sta1[:,0],X_som_sta1[:,1], c='w', marker='o', s=5) 
        plt.title('sta1')
        plt.xlabel('%s')%x
        plt.ylabel('%s')%y
        #stations 2
    if sta==2:
        X_eff_sta2 = X_eff[:, n_attind:n_attind*2]
        X_som_sta2 = X_som[:, n_attind:n_attind*2]
    

        # K
    fig, axes= plt.subplots(4, 2)
    
    plt.subplot(421)
    plt.scatter(X_eff_sta1[:,0],X_eff_sta1[:,1], c='r', marker='x') #K_Dur eff
    plt.scatter(X_som_sta1[:,0],X_som_sta1[:,1], c='w', marker='o', s=5) #K_Dur som
    plt.title('sta1')
    plt.subplot(423)
    plt.scatter(X_eff_sta1[:,0],X_eff_sta1[:,2], c='r', marker='x') #K_cent_f
    plt.scatter(X_som_sta1[:,0],X_som_sta1[:,2], c='w', marker='o', s=5)
    plt.subplot(425)
    plt.scatter(X_eff_sta1[:,0],X_eff_sta1[:,3], c='r', marker='x') #K_A
    plt.scatter(X_som_sta1[:,0],X_som_sta1[:,3], c='w', marker='o', s=5)
    plt.subplot(427)
    plt.scatter(X_eff_sta1[:,0],X_eff_sta1[:,4], c='r', marker='x') #K_Dur/A
    plt.scatter(X_som_sta1[:,0],X_som_sta1[:,4], c='w', marker='o', s=5)
    
    #station 2 colonne 2
    plt.subplot(422)
    plt.scatter(X_eff_sta2[:,0],X_eff_sta2[:,1], c='r', marker='x')
    plt.scatter(X_som_sta2[:,0],X_som_sta2[:,1], c='w', marker='o', s=5)
    plt.title('sta2')
    plt.subplot(424)
    plt.scatter(X_eff_sta2[:,0],X_eff_sta2[:,2], c='r', marker='x')
    plt.scatter(X_som_sta2[:,0],X_som_sta2[:,2], c='w', marker='o', s=5)
    plt.subplot(426)
    plt.scatter(X_eff_sta2[:,0],X_eff_sta2[:,3], c='r', marker='x')
    plt.scatter(X_som_sta2[:,0],X_som_sta2[:,3], c='w', marker='o', s=5)
    plt.subplot(428)
    plt.scatter(X_eff_sta2[:,0],X_eff_sta2[:,4], c='r', marker='x')
    plt.scatter(X_som_sta2[:,0],X_som_sta2[:,4], c='w', marker='o', s=5)
    
    
    fig.savefig('K1.png')
    
    fig, axes= plt.subplots(4, 2)
    
    plt.subplot(421)
    plt.scatter(X_eff_sta1[:,0],X_eff_sta1[:,5], c='r', marker='x') #K_Dur eff
    plt.scatter(X_som_sta1[:,0],X_som_sta1[:,5], c='w', marker='o', s=5) #K_Dur som
    plt.title('sta1')
    plt.subplot(423)
    plt.scatter(X_eff_sta1[:,0],X_eff_sta1[:,6], c='r', marker='x') #K_cent_f
    plt.scatter(X_som_sta1[:,0],X_som_sta1[:,6], c='w', marker='o', s=5)
    plt.subplot(425)
    plt.scatter(X_eff_sta1[:,0],X_eff_sta1[:,7], c='r', marker='x') #K_A
    plt.scatter(X_som_sta1[:,0],X_som_sta1[:,7], c='w', marker='o', s=5)
    #plt.subplot(427)
    #plt.scatter(X_eff_sta1[:,0],X_eff_sta1[:,8], c='r', marker='x') #K_Dur/A
    #plt.scatter(X_som_sta1[:,0],X_som_sta1[:,8], c='w', marker='o', s=5)
    
    #station 2 colonne 2
    plt.subplot(422)
    plt.scatter(X_eff_sta2[:,0],X_eff_sta2[:,5], c='r', marker='x')
    plt.scatter(X_som_sta2[:,0],X_som_sta2[:,5], c='w', marker='o', s=5)
    plt.title('sta2')
    plt.subplot(424)
    plt.scatter(X_eff_sta2[:,0],X_eff_sta2[:,6], c='r', marker='x')
    plt.scatter(X_som_sta2[:,0],X_som_sta2[:,6], c='w', marker='o', s=5)
    plt.subplot(426)
    plt.scatter(X_eff_sta2[:,0],X_eff_sta2[:,7], c='r', marker='x')
    plt.scatter(X_som_sta2[:,0],X_som_sta2[:,7], c='w', marker='o', s=5)
    plt.subplot(428) # manque AsDec donc un graphe en moins
    #plt.scatter(X_eff_sta2[:,0],X_eff_sta2[:,8], c='r', marker='x')
    #plt.scatter(X_som_sta2[:,0],X_som_sta2[:,8], c='w', marker='o', s=5)
    
    
    fig.savefig('K2.png')
"""
    
def plot_class_all(X_eff, X_som, n_attind):
    """
    plot all the graph 1vs1 for the 8 attributs
    for the two station
    """
    #station 1
    X_eff_sta1 = X_eff[:,0:n_attind]  # les colonnes suivent l'ordres d'entrée des attributs dans l'indexe
    X_som_sta1 = X_som[:,0:n_attind]
    
    #scaler = MinMaxScaler().fit(X_eff_sta1)
    #scaler = MinMaxScaler().fit(X_som_sta1)
    
    #stations 2
    
    X_eff_sta2 = X_eff[:, n_attind:n_attind*2]
    X_som_sta2 = X_som[:, n_attind:n_attind*2]
    
    #scaler = MinMaxScaler().fit(X_eff_sta2)
    #scaler = MinMaxScaler().fit(X_som_sta2)

   # K
    fig, axes= plt.subplots(4, 2)
    
    plt.subplot(421)
    plt.scatter(X_eff_sta1[:,0],X_eff_sta1[:,1], c='r', marker='x') #K_Dur eff
    plt.scatter(X_som_sta1[:,0],X_som_sta1[:,1], c='w', marker='o', s=5) #K_Dur som
    plt.xlabel('K')
    plt.ylabel('Dur')
    plt.title('sta1')
    plt.subplot(423)
    plt.scatter(X_eff_sta1[:,0],X_eff_sta1[:,2], c='r', marker='x') #K_cent_f
    plt.scatter(X_som_sta1[:,0],X_som_sta1[:,2], c='w', marker='o', s=5)
    plt.xlabel('K')
    plt.ylabel('cent_f')
    plt.subplot(425)
    plt.scatter(X_eff_sta1[:,0],X_eff_sta1[:,3], c='r', marker='x') #K_A
    plt.scatter(X_som_sta1[:,0],X_som_sta1[:,3], c='w', marker='o', s=5)
    plt.xlabel('K')
    plt.ylabel('A')
    plt.subplot(427)
    plt.scatter(X_eff_sta1[:,0],X_eff_sta1[:,4], c='r', marker='x') #K_Dur/A
    plt.scatter(X_som_sta1[:,0],X_som_sta1[:,4], c='w', marker='o', s=5)
    plt.xlabel('K')
    plt.ylabel('Dur/A')
    
    #station 2 colonne 2
    plt.subplot(422)
    plt.scatter(X_eff_sta2[:,0],X_eff_sta2[:,1], c='r', marker='x')
    plt.scatter(X_som_sta2[:,0],X_som_sta2[:,1], c='w', marker='o', s=5)
    plt.xlabel('K')
    plt.ylabel('Dur')
    plt.title('sta2')
    plt.subplot(424)
    plt.scatter(X_eff_sta2[:,0],X_eff_sta2[:,2], c='r', marker='x')
    plt.scatter(X_som_sta2[:,0],X_som_sta2[:,2], c='w', marker='o', s=5)
    plt.xlabel('K')
    plt.ylabel('cent_f')
    plt.subplot(426)
    plt.scatter(X_eff_sta2[:,0],X_eff_sta2[:,3], c='r', marker='x')
    plt.scatter(X_som_sta2[:,0],X_som_sta2[:,3], c='w', marker='o', s=5)
    plt.xlabel('K')
    plt.ylabel('A')
    plt.subplot(428)
    plt.scatter(X_eff_sta2[:,0],X_eff_sta2[:,4], c='r', marker='x')
    plt.scatter(X_som_sta2[:,0],X_som_sta2[:,4], c='w', marker='o', s=5)
    plt.xlabel('K')
    plt.ylabel('Dur/A')
    
    
    fig.savefig('K1.png')
    
    fig, axes= plt.subplots(4, 2)
    
    plt.subplot(421)
    plt.scatter(X_eff_sta1[:,0],X_eff_sta1[:,5], c='r', marker='x') 
    plt.scatter(X_som_sta1[:,0],X_som_sta1[:,5], c='w', marker='o', s=5) 
    plt.xlabel('K')
    plt.ylabel('dom_f')
    plt.title('sta1')
    plt.subplot(423)
    plt.scatter(X_eff_sta1[:,0],X_eff_sta1[:,6], c='r', marker='x') 
    plt.scatter(X_som_sta1[:,0],X_som_sta1[:,6], c='w', marker='o', s=5)
    plt.xlabel('K')
    plt.ylabel('E')
    plt.subplot(425)
    plt.scatter(X_eff_sta1[:,0],X_eff_sta1[:,7], c='r', marker='x') 
    plt.scatter(X_som_sta1[:,0],X_som_sta1[:,7], c='w', marker='o', s=5)
    plt.xlabel('K')
    plt.ylabel('maxA_mean')
    plt.subplot(427)
    #plt.scatter(X_eff_sta1[:,0],X_eff_sta1[:,8], c='r', marker='x') 
    #plt.scatter(X_som_sta1[:,0],X_som_sta1[:,8], c='w', marker='o', s=5)
    plt.xlabel('K')
    plt.ylabel('AsDec')
    
    #station 2 colonne 2
    plt.subplot(422)
    plt.scatter(X_eff_sta2[:,0],X_eff_sta2[:,5], c='r', marker='x')
    plt.scatter(X_som_sta2[:,0],X_som_sta2[:,5], c='w', marker='o', s=5)
    plt.xlabel('K')
    plt.ylabel('dom_f')
    plt.title('sta2')
    plt.subplot(424)
    plt.scatter(X_eff_sta2[:,0],X_eff_sta2[:,6], c='r', marker='x')
    plt.scatter(X_som_sta2[:,0],X_som_sta2[:,6], c='w', marker='o', s=5)
    plt.xlabel('K')
    plt.ylabel('E')
    plt.subplot(426)
    plt.scatter(X_eff_sta2[:,0],X_eff_sta2[:,7], c='r', marker='x')
    plt.scatter(X_som_sta2[:,0],X_som_sta2[:,7], c='w', marker='o', s=5)
    plt.xlabel('K')
    plt.ylabel('maxA_mean')
    plt.subplot(428) # manque AsDec donc un graphe en moins
    #plt.scatter(X_eff_sta2[:,0],X_eff_sta2[:,8], c='r', marker='x')
    #plt.scatter(X_som_sta2[:,0],X_som_sta2[:,8], c='w', marker='o', s=5)
    plt.xlabel('K')
    plt.ylabel('AsDec')
    
    
    fig.savefig('K2.png')
    
    # Dur
    fig, axes= plt.subplots(4, 2)
    
    plt.subplot(421)
    plt.scatter(X_eff_sta1[:,1],X_eff_sta1[:,0], c='r', marker='x') #K_Dur eff
    plt.scatter(X_som_sta1[:,1],X_som_sta1[:,0], c='w', marker='o', s=5) #K_Dur som
    plt.xlabel('Dur')
    plt.ylabel('K')
    plt.title('sta1')
    plt.subplot(423)
    plt.scatter(X_eff_sta1[:,1],X_eff_sta1[:,2], c='r', marker='x') #K_cent_f
    plt.scatter(X_som_sta1[:,1],X_som_sta1[:,2], c='w', marker='o', s=5)
    plt.xlabel('Dur')
    plt.ylabel('cent_f')
    plt.subplot(425)
    plt.scatter(X_eff_sta1[:,1],X_eff_sta1[:,3], c='r', marker='x') #K_A
    plt.scatter(X_som_sta1[:,1],X_som_sta1[:,3], c='w', marker='o', s=5)
    plt.xlabel('Dur')
    plt.ylabel('A')
    plt.subplot(427)
    plt.scatter(X_eff_sta1[:,1],X_eff_sta1[:,4], c='r', marker='x') #K_Dur/A
    plt.scatter(X_som_sta1[:,1],X_som_sta1[:,4], c='w', marker='o', s=5)
    plt.xlabel('Dur')
    plt.ylabel('Dur/A')
    
    #station 2 colonne 2
    plt.subplot(422)
    plt.scatter(X_eff_sta2[:,1],X_eff_sta2[:,0], c='r', marker='x')
    plt.scatter(X_som_sta2[:,1],X_som_sta2[:,0], c='w', marker='o', s=5)
    plt.xlabel('Dur')
    plt.ylabel('K')
    plt.title('sta2')
    plt.subplot(424)
    plt.scatter(X_eff_sta2[:,1],X_eff_sta2[:,2], c='r', marker='x')
    plt.scatter(X_som_sta2[:,1],X_som_sta2[:,2], c='w', marker='o', s=5)
    plt.xlabel('Dur')
    plt.ylabel('cent_f')
    plt.subplot(426)
    plt.scatter(X_eff_sta2[:,1],X_eff_sta2[:,3], c='r', marker='x')
    plt.scatter(X_som_sta2[:,1],X_som_sta2[:,3], c='w', marker='o', s=5)
    plt.xlabel('Dur')
    plt.ylabel('A')
    plt.subplot(428)
    plt.scatter(X_eff_sta2[:,1],X_eff_sta2[:,4], c='r', marker='x')
    plt.scatter(X_som_sta2[:,1],X_som_sta2[:,4], c='w', marker='o', s=5)
    plt.xlabel('Dur')
    plt.ylabel('Dur/A')
    
    
    fig.savefig('Dur1.png')
    
    fig, axes= plt.subplots(4, 2)
    
    plt.subplot(421)
    plt.scatter(X_eff_sta1[:,1],X_eff_sta1[:,5], c='r', marker='x') #K_Dur eff
    plt.scatter(X_som_sta1[:,1],X_som_sta1[:,5], c='w', marker='o', s=5) #K_Dur som
    plt.xlabel('Dur')
    plt.ylabel('dom_f')
    plt.title('sta1')
    plt.subplot(423)
    plt.scatter(X_eff_sta1[:,1],X_eff_sta1[:,6], c='r', marker='x') #K_cent_f
    plt.scatter(X_som_sta1[:,1],X_som_sta1[:,6], c='w', marker='o', s=5)
    plt.xlabel('Dur')
    plt.ylabel('E')
    plt.subplot(425)
    plt.scatter(X_eff_sta1[:,1],X_eff_sta1[:,7], c='r', marker='x') #K_A
    plt.scatter(X_som_sta1[:,1],X_som_sta1[:,7], c='w', marker='o', s=5)
    plt.xlabel('Dur')
    plt.ylabel('maxA_mean')
    plt.subplot(427)
    #plt.scatter(X_eff_sta1[:,1],X_eff_sta1[:,8], c='r', marker='x') #K_Dur/A
    #plt.scatter(X_som_sta1[:,1],X_som_sta1[:,8], c='w', marker='o', s=5)
    plt.xlabel('Dur')
    plt.ylabel('AsDec')
    
    #station 2 colonne 2
    plt.subplot(422)
    plt.scatter(X_eff_sta2[:,1],X_eff_sta2[:,5], c='r', marker='x')
    plt.scatter(X_som_sta2[:,1],X_som_sta2[:,5], c='w', marker='o', s=5)
    plt.xlabel('Dur')
    plt.ylabel('dom_f')
    plt.title('sta2')
    plt.subplot(424)
    plt.scatter(X_eff_sta2[:,1],X_eff_sta2[:,6], c='r', marker='x')
    plt.scatter(X_som_sta2[:,1],X_som_sta2[:,6], c='w', marker='o', s=5)
    plt.xlabel('Dur')
    plt.ylabel('E')
    plt.subplot(426)
    plt.scatter(X_eff_sta2[:,1],X_eff_sta2[:,7], c='r', marker='x')
    plt.scatter(X_som_sta2[:,1],X_som_sta2[:,7], c='w', marker='o', s=5)
    plt.xlabel('Dur')
    plt.ylabel('maxA_mean')
    plt.subplot(428)
    #plt.scatter(X_eff_sta2[:,1],X_eff_sta2[:,8], c='r', marker='x')
    #plt.scatter(X_som_sta2[:,1],X_som_sta2[:,8], c='w', marker='o', s=5)
    plt.xlabel('Dur')
    plt.ylabel('AsDec')
    
    
    fig.savefig('Dur2.png')
    
    # cent_f
    fig, axes= plt.subplots(4, 2)
    
    plt.subplot(421)
    plt.scatter(X_eff_sta1[:,2],X_eff_sta1[:,0], c='r', marker='x') #K_Dur eff
    plt.scatter(X_som_sta1[:,2],X_som_sta1[:,0], c='w', marker='o', s=5) #K_Dur som
    plt.xlabel('cent_f')
    plt.ylabel('K')
    plt.title('sta1')
    plt.subplot(423)
    plt.scatter(X_eff_sta1[:,2],X_eff_sta1[:,1], c='r', marker='x') #K_cent_f
    plt.scatter(X_som_sta1[:,2],X_som_sta1[:,1], c='w', marker='o', s=5)
    plt.xlabel('cent_f')
    plt.ylabel('Dur')
    plt.subplot(425)
    plt.scatter(X_eff_sta1[:,2],X_eff_sta1[:,3], c='r', marker='x') #K_A
    plt.scatter(X_som_sta1[:,2],X_som_sta1[:,3], c='w', marker='o', s=5)
    plt.xlabel('cent_f')
    plt.ylabel('A')
    plt.subplot(427)
    plt.scatter(X_eff_sta1[:,2],X_eff_sta1[:,4], c='r', marker='x') #K_Dur/A
    plt.scatter(X_som_sta1[:,2],X_som_sta1[:,4], c='w', marker='o', s=5)
    plt.xlabel('cent_f')
    plt.ylabel('Dur/A')
    
    #station 2 colonne 2
    plt.subplot(422)
    plt.scatter(X_eff_sta2[:,2],X_eff_sta2[:,0], c='r', marker='x')
    plt.scatter(X_som_sta2[:,2],X_som_sta2[:,0], c='w', marker='o', s=5)
    plt.xlabel('cent_f')
    plt.ylabel('K')
    plt.title('sta2')
    plt.subplot(424)
    plt.scatter(X_eff_sta2[:,2],X_eff_sta2[:,1], c='r', marker='x')
    plt.scatter(X_som_sta2[:,2],X_som_sta2[:,1], c='w', marker='o', s=5)
    plt.xlabel('cent_f')
    plt.ylabel('Dur')
    plt.subplot(426)
    plt.scatter(X_eff_sta2[:,2],X_eff_sta2[:,3], c='r', marker='x')
    plt.scatter(X_som_sta2[:,2],X_som_sta2[:,3], c='w', marker='o', s=5)
    plt.xlabel('cent_f')
    plt.ylabel('A')
    plt.subplot(428)
    plt.scatter(X_eff_sta2[:,2],X_eff_sta2[:,4], c='r', marker='x')
    plt.scatter(X_som_sta2[:,2],X_som_sta2[:,4], c='w', marker='o', s=5)
    plt.xlabel('cent_f')
    plt.ylabel('Dur/A')
    
    fig.savefig('cent_f1.png')
    
    fig, axes= plt.subplots(4, 2)
    
    plt.subplot(421)
    plt.scatter(X_eff_sta1[:,2],X_eff_sta1[:,5], c='r', marker='x') #K_Dur eff
    plt.scatter(X_som_sta1[:,2],X_som_sta1[:,5], c='w', marker='o', s=5) #K_Dur som
    plt.xlabel('cent_f')
    plt.ylabel('dom_f')
    plt.title('sta1')
    plt.subplot(423)
    plt.scatter(X_eff_sta1[:,2],X_eff_sta1[:,6], c='r', marker='x') #K_cent_f
    plt.scatter(X_som_sta1[:,2],X_som_sta1[:,6], c='w', marker='o', s=5)
    plt.xlabel('cent_f')
    plt.ylabel('E')
    plt.subplot(425)
    plt.scatter(X_eff_sta1[:,2],X_eff_sta1[:,7], c='r', marker='x') #K_A
    plt.scatter(X_som_sta1[:,2],X_som_sta1[:,7], c='w', marker='o', s=5)
    plt.xlabel('cent_f')
    plt.ylabel('maxA_mean')
    plt.subplot(427)
    #plt.scatter(X_eff_sta1[:,2],X_eff_sta1[:,8], c='r', marker='x') #K_Dur/A
    #plt.scatter(X_som_sta1[:,2],X_som_sta1[:,8], c='w', marker='o', s=5)
    plt.xlabel('cent_f')
    plt.ylabel('AsDec')
    
    #station 2 colonne 2
    plt.subplot(422)
    plt.scatter(X_eff_sta2[:,2],X_eff_sta2[:,5], c='r', marker='x')
    plt.scatter(X_som_sta2[:,2],X_som_sta2[:,5], c='w', marker='o', s=5)
    plt.xlabel('cent_f')
    plt.ylabel('dom_f')
    plt.title('sta2')
    plt.subplot(424)
    plt.scatter(X_eff_sta2[:,2],X_eff_sta2[:,6], c='r', marker='x')
    plt.scatter(X_som_sta2[:,2],X_som_sta2[:,6], c='w', marker='o', s=5)
    plt.xlabel('cent_f')
    plt.ylabel('E')
    plt.subplot(426)
    plt.scatter(X_eff_sta2[:,2],X_eff_sta2[:,7], c='r', marker='x')
    plt.scatter(X_som_sta2[:,2],X_som_sta2[:,7], c='w', marker='o', s=5)
    plt.xlabel('cent_f')
    plt.ylabel('maxA_mean')
    plt.subplot(428)
    #plt.scatter(X_eff_sta2[:,2],X_eff_sta2[:,8], c='r', marker='x')
    #plt.scatter(X_som_sta2[:,2],X_som_sta2[:,8], c='w', marker='o', s=5)
    plt.xlabel('cent_f')
    plt.ylabel('AsDec')
    
    
    fig.savefig('cent_f2.png')
    
    # A
    fig, axes= plt.subplots(4, 2)
    
    plt.subplot(421)
    plt.scatter(X_eff_sta1[:,3],X_eff_sta1[:,0], c='r', marker='x') #K_Dur eff
    plt.scatter(X_som_sta1[:,3],X_som_sta1[:,0], c='w', marker='o', s=5) #K_Dur som
    plt.xlabel('A')
    plt.ylabel('K')
    plt.title('sta1')
    plt.subplot(423)
    plt.scatter(X_eff_sta1[:,3],X_eff_sta1[:,1], c='r', marker='x') #K_cent_f
    plt.scatter(X_som_sta1[:,3],X_som_sta1[:,1], c='w', marker='o', s=5)
    plt.xlabel('A')
    plt.ylabel('Dur')
    plt.subplot(425)
    plt.scatter(X_eff_sta1[:,3],X_eff_sta1[:,2], c='r', marker='x') #K_A
    plt.scatter(X_som_sta1[:,3],X_som_sta1[:,2], c='w', marker='o', s=5)
    plt.xlabel('A')
    plt.ylabel('cent_f')
    plt.subplot(427)
    plt.scatter(X_eff_sta1[:,3],X_eff_sta1[:,4], c='r', marker='x') #K_Dur/A
    plt.scatter(X_som_sta1[:,3],X_som_sta1[:,4], c='w', marker='o', s=5)
    plt.xlabel('A')
    plt.ylabel('Dur/A')
    
    #station 2 colonne 2
    plt.subplot(422)
    plt.scatter(X_eff_sta2[:,3],X_eff_sta2[:,0], c='r', marker='x')
    plt.scatter(X_som_sta2[:,3],X_som_sta2[:,0], c='w', marker='o', s=5)
    plt.xlabel('A')
    plt.ylabel('K')
    plt.title('sta2')
    plt.subplot(424)
    plt.scatter(X_eff_sta2[:,3],X_eff_sta2[:,1], c='r', marker='x')
    plt.scatter(X_som_sta2[:,3],X_som_sta2[:,1], c='w', marker='o', s=5)
    plt.xlabel('A')
    plt.ylabel('Dur')
    plt.subplot(426)
    plt.scatter(X_eff_sta2[:,3],X_eff_sta2[:,2], c='r', marker='x')
    plt.scatter(X_som_sta2[:,3],X_som_sta2[:,2], c='w', marker='o', s=5)
    plt.xlabel('A')
    plt.ylabel('cent_f')
    plt.subplot(428)
    plt.scatter(X_eff_sta2[:,3],X_eff_sta2[:,4], c='r', marker='x')
    plt.scatter(X_som_sta2[:,3],X_som_sta2[:,4], c='w', marker='o', s=5)
    plt.xlabel('A')
    plt.ylabel('Dur/A')
    
    
    fig.savefig('A1.png')
    
    fig, axes= plt.subplots(4, 2)
    
    plt.subplot(421)
    plt.scatter(X_eff_sta1[:,3],X_eff_sta1[:,5], c='r', marker='x') #K_Dur eff
    plt.scatter(X_som_sta1[:,3],X_som_sta1[:,5], c='w', marker='o', s=5) #K_Dur som
    plt.xlabel('A')
    plt.ylabel('dom_f')
    plt.title('sta1')
    plt.subplot(423)
    plt.scatter(X_eff_sta1[:,3],X_eff_sta1[:,6], c='r', marker='x') #K_cent_f
    plt.scatter(X_som_sta1[:,3],X_som_sta1[:,6], c='w', marker='o', s=5)
    plt.xlabel('A')
    plt.ylabel('E')
    plt.subplot(425)
    plt.scatter(X_eff_sta1[:,3],X_eff_sta1[:,7], c='r', marker='x') #K_A
    plt.scatter(X_som_sta1[:,3],X_som_sta1[:,7], c='w', marker='o', s=5)
    plt.xlabel('A')
    plt.ylabel('maxA_mean')
    plt.subplot(427)
    #plt.scatter(X_eff_sta1[:,3],X_eff_sta1[:,8], c='r', marker='x') #K_Dur/A
    #plt.scatter(X_som_sta1[:,3],X_som_sta1[:,8], c='w', marker='o', s=5)
    plt.xlabel('A')
    plt.ylabel('AsDec')
    
    #station 2 colonne 2
    plt.subplot(422)
    plt.scatter(X_eff_sta2[:,3],X_eff_sta2[:,5], c='r', marker='x')
    plt.scatter(X_som_sta2[:,3],X_som_sta2[:,5], c='w', marker='o', s=5)
    plt.xlabel('A')
    plt.ylabel('dom_f')
    plt.title('sta2')
    plt.subplot(424)
    plt.scatter(X_eff_sta2[:,3],X_eff_sta2[:,6], c='r', marker='x')
    plt.scatter(X_som_sta2[:,3],X_som_sta2[:,6], c='w', marker='o', s=5)
    plt.xlabel('A')
    plt.ylabel('E')
    plt.subplot(426)
    plt.scatter(X_eff_sta2[:,3],X_eff_sta2[:,7], c='r', marker='x')
    plt.scatter(X_som_sta2[:,3],X_som_sta2[:,7], c='w', marker='o', s=5)
    plt.xlabel('A')
    plt.ylabel('maxA_mean')
    plt.subplot(428)
    #plt.scatter(X_eff_sta2[:,3],X_eff_sta2[:,8], c='r', marker='x')
    #plt.scatter(X_som_sta2[:,3],X_som_sta2[:,8], c='w', marker='o', s=5)
    plt.xlabel('A')
    plt.ylabel('AsDec')
    
    
    fig.savefig('A2.png')
    
    # dur/A
    fig, axes= plt.subplots(4, 2)
    
    plt.subplot(421)
    plt.scatter(X_eff_sta1[:,4],X_eff_sta1[:,0], c='r', marker='x') #K_Dur eff
    plt.scatter(X_som_sta1[:,4],X_som_sta1[:,0], c='w', marker='o', s=5) #K_Dur som
    plt.xlabel('dur/A')
    plt.ylabel('K')
    plt.title('sta1')
    plt.subplot(423)
    plt.scatter(X_eff_sta1[:,4],X_eff_sta1[:,1], c='r', marker='x') #K_cent_f
    plt.scatter(X_som_sta1[:,4],X_som_sta1[:,1], c='w', marker='o', s=5)
    plt.xlabel('dur/A')
    plt.ylabel('Dur')
    plt.subplot(425)
    plt.scatter(X_eff_sta1[:,4],X_eff_sta1[:,2], c='r', marker='x') #K_A
    plt.scatter(X_som_sta1[:,4],X_som_sta1[:,2], c='w', marker='o', s=5)
    plt.xlabel('dur/A')
    plt.ylabel('cent_f')
    plt.subplot(427)
    plt.scatter(X_eff_sta1[:,4],X_eff_sta1[:,3], c='r', marker='x') #K_Dur/A
    plt.scatter(X_som_sta1[:,4],X_som_sta1[:,3], c='w', marker='o', s=5)
    plt.xlabel('dur/A')
    plt.ylabel('A')
    
    #station 2 colonne 2
    plt.subplot(422)
    plt.scatter(X_eff_sta2[:,4],X_eff_sta2[:,0], c='r', marker='x')
    plt.scatter(X_som_sta2[:,4],X_som_sta2[:,0], c='w', marker='o', s=5)
    plt.xlabel('dur/A')
    plt.ylabel('K')
    plt.title('sta2')
    plt.subplot(424)
    plt.scatter(X_eff_sta2[:,4],X_eff_sta2[:,1], c='r', marker='x')
    plt.scatter(X_som_sta2[:,4],X_som_sta2[:,1], c='w', marker='o', s=5)
    plt.xlabel('dur/A')
    plt.ylabel('Dur')
    plt.subplot(426)
    plt.scatter(X_eff_sta2[:,4],X_eff_sta2[:,2], c='r', marker='x')
    plt.scatter(X_som_sta2[:,4],X_som_sta2[:,2], c='w', marker='o', s=5)
    plt.xlabel('dur/A')
    plt.ylabel('cent_f')
    plt.subplot(428)
    plt.scatter(X_eff_sta2[:,4],X_eff_sta2[:,3], c='r', marker='x')
    plt.scatter(X_som_sta2[:,4],X_som_sta2[:,3], c='w', marker='o', s=5)
    plt.xlabel('dur/A')
    plt.ylabel('A')
    
    
    fig.savefig('Dur_A1.png')
    
    fig, axes= plt.subplots(4, 2)
    
    plt.subplot(421)
    plt.scatter(X_eff_sta1[:,4],X_eff_sta1[:,5], c='r', marker='x') #K_Dur eff
    plt.scatter(X_som_sta1[:,4],X_som_sta1[:,5], c='w', marker='o', s=5) #K_Dur som
    plt.xlabel('dur/A')
    plt.ylabel('dom_f')
    plt.title('sta1')
    plt.subplot(423)
    plt.scatter(X_eff_sta1[:,4],X_eff_sta1[:,6], c='r', marker='x') #K_cent_f
    plt.scatter(X_som_sta1[:,4],X_som_sta1[:,6], c='w', marker='o', s=5)
    plt.xlabel('dur/A')
    plt.ylabel('E')
    plt.subplot(425)
    plt.scatter(X_eff_sta1[:,4],X_eff_sta1[:,7], c='r', marker='x') #K_A
    plt.scatter(X_som_sta1[:,4],X_som_sta1[:,7], c='w', marker='o', s=5)
    plt.xlabel('dur/A')
    plt.ylabel('maxA_mean')
    plt.subplot(427)
    #plt.scatter(X_eff_sta1[:,4],X_eff_sta1[:,8], c='r', marker='x') #K_Dur/A
    #plt.scatter(X_som_sta1[:,4],X_som_sta1[:,8], c='w', marker='o', s=5)
    plt.xlabel('dur/A')
    plt.ylabel('AsDec')
    
    #station 2 colonne 2
    plt.subplot(422)
    plt.scatter(X_eff_sta2[:,4],X_eff_sta2[:,5], c='r', marker='x')
    plt.scatter(X_som_sta2[:,4],X_som_sta2[:,5], c='w', marker='o', s=5)
    plt.xlabel('dur/A')
    plt.ylabel('dom_f')
    plt.title('sta2')
    plt.subplot(424)
    plt.scatter(X_eff_sta2[:,4],X_eff_sta2[:,6], c='r', marker='x')
    plt.scatter(X_som_sta2[:,4],X_som_sta2[:,6], c='w', marker='o', s=5)
    plt.xlabel('dur/A')
    plt.ylabel('E')
    plt.subplot(426)
    plt.scatter(X_eff_sta2[:,4],X_eff_sta2[:,7], c='r', marker='x')
    plt.scatter(X_som_sta2[:,4],X_som_sta2[:,7], c='w', marker='o', s=5)
    plt.xlabel('dur/A')
    plt.ylabel('maxA_mean')
    plt.subplot(428)
    #plt.scatter(X_eff_sta2[:,4],X_eff_sta2[:,8], c='r', marker='x')
    #plt.scatter(X_som_sta2[:,4],X_som_sta2[:,8], c='w', marker='o', s=5)
    plt.xlabel('dur/A')
    plt.ylabel('AsDec')
    
    
    fig.savefig('Dur_A2.png')
    
    # dom_f
    fig, axes= plt.subplots(4, 2)
    
    plt.subplot(421)
    plt.scatter(X_eff_sta1[:,5],X_eff_sta1[:,0], c='r', marker='x') #K_Dur eff
    plt.scatter(X_som_sta1[:,5],X_som_sta1[:,0], c='w', marker='o', s=5) #K_Dur som
    plt.xlabel('dom_f')
    plt.ylabel('K')
    plt.title('sta1')
    plt.subplot(423)
    plt.scatter(X_eff_sta1[:,5],X_eff_sta1[:,1], c='r', marker='x') #K_cent_f
    plt.scatter(X_som_sta1[:,5],X_som_sta1[:,1], c='w', marker='o', s=5)
    plt.xlabel('dom_f')
    plt.ylabel('Dur')
    plt.subplot(425)
    plt.scatter(X_eff_sta1[:,5],X_eff_sta1[:,2], c='r', marker='x') #K_A
    plt.scatter(X_som_sta1[:,5],X_som_sta1[:,2], c='w', marker='o', s=5)
    plt.xlabel('dom_f')
    plt.ylabel('cent_f')
    plt.subplot(427)
    plt.scatter(X_eff_sta1[:,5],X_eff_sta1[:,3], c='r', marker='x') #K_Dur/A
    plt.scatter(X_som_sta1[:,5],X_som_sta1[:,3], c='w', marker='o', s=5)
    plt.xlabel('dom_f')
    plt.ylabel('A')
    
    #station 2 colonne 2
    plt.subplot(422)
    plt.scatter(X_eff_sta2[:,5],X_eff_sta2[:,0], c='r', marker='x')
    plt.scatter(X_som_sta2[:,5],X_som_sta2[:,0], c='w', marker='o', s=5)
    plt.xlabel('dom_f')
    plt.ylabel('K')
    plt.title('sta2')
    plt.subplot(424)
    plt.scatter(X_eff_sta2[:,5],X_eff_sta2[:,1], c='r', marker='x')
    plt.scatter(X_som_sta2[:,5],X_som_sta2[:,1], c='w', marker='o', s=5)
    plt.xlabel('dom_f')
    plt.ylabel('Dur')
    plt.subplot(426)
    plt.scatter(X_eff_sta2[:,5],X_eff_sta2[:,2], c='r', marker='x')
    plt.scatter(X_som_sta2[:,5],X_som_sta2[:,2], c='w', marker='o', s=5)
    plt.xlabel('dom_f')
    plt.ylabel('cent_f')
    plt.subplot(428)
    plt.scatter(X_eff_sta2[:,5],X_eff_sta2[:,3], c='r', marker='x')
    plt.scatter(X_som_sta2[:,5],X_som_sta2[:,3], c='w', marker='o', s=5)
    plt.xlabel('dom_f')
    plt.ylabel('A')
    
    
    fig.savefig('dom_f1.png')
    
    fig, axes= plt.subplots(4, 2)
    
    plt.subplot(421)
    plt.scatter(X_eff_sta1[:,5],X_eff_sta1[:,4], c='r', marker='x') #K_Dur eff
    plt.scatter(X_som_sta1[:,5],X_som_sta1[:,4], c='w', marker='o', s=5) #K_Dur som
    plt.xlabel('dom_f')
    plt.ylabel('Dur/A')
    plt.title('sta1')
    plt.subplot(423)
    plt.scatter(X_eff_sta1[:,5],X_eff_sta1[:,6], c='r', marker='x') #K_cent_f
    plt.scatter(X_som_sta1[:,5],X_som_sta1[:,6], c='w', marker='o', s=5)
    plt.xlabel('dom_f')
    plt.ylabel('E')
    plt.subplot(425)
    plt.scatter(X_eff_sta1[:,5],X_eff_sta1[:,7], c='r', marker='x') #K_A
    plt.scatter(X_som_sta1[:,5],X_som_sta1[:,7], c='w', marker='o', s=5)
    plt.xlabel('dom_f')
    plt.ylabel('maxA_mean')
    plt.subplot(427)
    #plt.scatter(X_eff_sta1[:,5],X_eff_sta1[:,8], c='r', marker='x') #K_Dur/A
    #plt.scatter(X_som_sta1[:,5],X_som_sta1[:,8], c='w', marker='o', s=5)
    plt.xlabel('dom_f')
    plt.ylabel('AsDec')
    
    #station 2 colonne 2
    plt.subplot(422)
    plt.scatter(X_eff_sta2[:,5],X_eff_sta2[:,4], c='r', marker='x')
    plt.scatter(X_som_sta2[:,5],X_som_sta2[:,4], c='w', marker='o', s=5)
    plt.xlabel('dom_f')
    plt.ylabel('Dur/A')
    plt.title('sta2')
    plt.subplot(424)
    plt.scatter(X_eff_sta2[:,5],X_eff_sta2[:,6], c='r', marker='x')
    plt.scatter(X_som_sta2[:,5],X_som_sta2[:,6], c='w', marker='o', s=5)
    plt.xlabel('dom_f')
    plt.ylabel('E')
    plt.subplot(426)
    plt.scatter(X_eff_sta2[:,5],X_eff_sta2[:,7], c='r', marker='x')
    plt.scatter(X_som_sta2[:,5],X_som_sta2[:,7], c='w', marker='o', s=5)
    plt.xlabel('dom_f')
    plt.ylabel('maxA_mean')
    plt.subplot(428)
    #plt.scatter(X_eff_sta2[:,5],X_eff_sta2[:,8], c='r', marker='x')
    #plt.scatter(X_som_sta2[:,5],X_som_sta2[:,8], c='w', marker='o', s=5)
    plt.xlabel('dom_f')
    plt.ylabel('AsDec')
    
    
    fig.savefig('dom_f2.png')
    
    # E
    fig, axes= plt.subplots(4, 2)
    
    plt.subplot(421)
    plt.scatter(X_eff_sta1[:,6],X_eff_sta1[:,0], c='r', marker='x') #K_Dur eff
    plt.scatter(X_som_sta1[:,6],X_som_sta1[:,0], c='w', marker='o', s=5) #K_Dur som
    plt.xlabel('E')
    plt.ylabel('K')
    plt.title('sta1')
    plt.subplot(423)
    plt.scatter(X_eff_sta1[:,6],X_eff_sta1[:,1], c='r', marker='x') #K_cent_f
    plt.scatter(X_som_sta1[:,6],X_som_sta1[:,1], c='w', marker='o', s=5)
    plt.xlabel('E')
    plt.ylabel('Dur')
    plt.subplot(425)
    plt.scatter(X_eff_sta1[:,6],X_eff_sta1[:,2], c='r', marker='x') #K_A
    plt.scatter(X_som_sta1[:,6],X_som_sta1[:,2], c='w', marker='o', s=5)
    plt.xlabel('E')
    plt.ylabel('cent_f')
    plt.subplot(427)
    plt.scatter(X_eff_sta1[:,6],X_eff_sta1[:,3], c='r', marker='x') #K_Dur/A
    plt.scatter(X_som_sta1[:,6],X_som_sta1[:,3], c='w', marker='o', s=5)
    plt.xlabel('E')
    plt.ylabel('A')
    
    #station 2 colonne 2
    plt.subplot(422)
    plt.scatter(X_eff_sta2[:,6],X_eff_sta2[:,0], c='r', marker='x')
    plt.scatter(X_som_sta2[:,6],X_som_sta2[:,0], c='w', marker='o', s=5)
    plt.xlabel('E')
    plt.ylabel('K')
    plt.title('sta2')
    plt.subplot(424)
    plt.scatter(X_eff_sta2[:,6],X_eff_sta2[:,1], c='r', marker='x')
    plt.scatter(X_som_sta2[:,6],X_som_sta2[:,1], c='w', marker='o', s=5)
    plt.xlabel('E')
    plt.ylabel('Dur')
    plt.subplot(426)
    plt.scatter(X_eff_sta2[:,6],X_eff_sta2[:,2], c='r', marker='x')
    plt.scatter(X_som_sta2[:,6],X_som_sta2[:,2], c='w', marker='o', s=5)
    plt.xlabel('E')
    plt.ylabel('cent_f')
    plt.subplot(428)
    plt.scatter(X_eff_sta2[:,6],X_eff_sta2[:,3], c='r', marker='x')
    plt.scatter(X_som_sta2[:,6],X_som_sta2[:,3], c='w', marker='o', s=5)
    plt.xlabel('E')
    plt.ylabel('A')
    
    
    fig.savefig('E1.png')
    
    fig, axes= plt.subplots(4, 2)
    
    plt.subplot(421)
    plt.scatter(X_eff_sta1[:,6],X_eff_sta1[:,4], c='r', marker='x') #K_Dur eff
    plt.scatter(X_som_sta1[:,6],X_som_sta1[:,4], c='w', marker='o', s=5) #K_Dur som
    plt.xlabel('E')
    plt.ylabel('Dur/A')
    plt.title('sta1')
    plt.subplot(423)
    plt.scatter(X_eff_sta1[:,6],X_eff_sta1[:,5], c='r', marker='x') #K_cent_f
    plt.scatter(X_som_sta1[:,6],X_som_sta1[:,5], c='w', marker='o', s=5)
    plt.xlabel('E')
    plt.ylabel('dom_f')
    plt.subplot(425)
    plt.scatter(X_eff_sta1[:,6],X_eff_sta1[:,7], c='r', marker='x') #K_A
    plt.scatter(X_som_sta1[:,6],X_som_sta1[:,7], c='w', marker='o', s=5)
    plt.xlabel('E')
    plt.ylabel('maxA_mean')
    plt.subplot(427)
    #plt.scatter(X_eff_sta1[:,6],X_eff_sta1[:,8], c='r', marker='x') #K_Dur/A
    #plt.scatter(X_som_sta1[:,6],X_som_sta1[:,8], c='w', marker='o', s=5)
    plt.xlabel('E')
    plt.ylabel('AsDec')
    
    #station 2 colonne 2
    plt.subplot(422)
    plt.scatter(X_eff_sta2[:,6],X_eff_sta2[:,4], c='r', marker='x')
    plt.scatter(X_som_sta2[:,6],X_som_sta2[:,4], c='w', marker='o', s=5)
    plt.xlabel('E')
    plt.ylabel('Dur/A')
    plt.title('sta2')
    plt.subplot(424)
    plt.scatter(X_eff_sta2[:,6],X_eff_sta2[:,5], c='r', marker='x')
    plt.scatter(X_som_sta2[:,6],X_som_sta2[:,5], c='w', marker='o', s=5)
    plt.xlabel('E')
    plt.ylabel('dom_f')
    plt.subplot(426)
    plt.scatter(X_eff_sta2[:,6],X_eff_sta2[:,7], c='r', marker='x')
    plt.scatter(X_som_sta2[:,6],X_som_sta2[:,7], c='w', marker='o', s=5)
    plt.xlabel('E')
    plt.ylabel('maxA_mean')
    plt.subplot(428)
    #plt.scatter(X_eff_sta2[:,6],X_eff_sta2[:,8], c='r', marker='x')
    #plt.scatter(X_som_sta2[:,6],X_som_sta2[:,8], c='w', marker='o', s=5)
    plt.xlabel('E')
    plt.ylabel('AsDec')
    
    
    fig.savefig('E2.png')
    
    # maxA_mean
    fig, axes= plt.subplots(4, 2)
    
    plt.subplot(421)
    plt.scatter(X_eff_sta1[:,7],X_eff_sta1[:,0], c='r', marker='x') #K_Dur eff
    plt.scatter(X_som_sta1[:,7],X_som_sta1[:,0], c='w', marker='o', s=5) #K_Dur som
    plt.xlabel('maxA_mean')
    plt.ylabel('K')
    plt.title('sta1')
    plt.subplot(423)
    plt.scatter(X_eff_sta1[:,7],X_eff_sta1[:,1], c='r', marker='x') #K_cent_f
    plt.scatter(X_som_sta1[:,7],X_som_sta1[:,1], c='w', marker='o', s=5)
    plt.xlabel('maxA_mean')
    plt.ylabel('Dur')
    plt.subplot(425)
    plt.scatter(X_eff_sta1[:,7],X_eff_sta1[:,2], c='r', marker='x') #K_A
    plt.scatter(X_som_sta1[:,7],X_som_sta1[:,2], c='w', marker='o', s=5)
    plt.xlabel('maxA_mean')
    plt.ylabel('cent_f')
    plt.subplot(427)
    plt.scatter(X_eff_sta1[:,7],X_eff_sta1[:,3], c='r', marker='x') #K_Dur/A
    plt.scatter(X_som_sta1[:,7],X_som_sta1[:,3], c='w', marker='o', s=5)
    plt.xlabel('maxA_mean')
    plt.ylabel('A')
    
    #station 2 colonne 2
    plt.subplot(422)
    plt.scatter(X_eff_sta2[:,7],X_eff_sta2[:,0], c='r', marker='x')
    plt.scatter(X_som_sta2[:,7],X_som_sta2[:,0], c='w', marker='o', s=5)
    plt.xlabel('maxA_mean')
    plt.ylabel('K')
    plt.title('sta2')
    plt.subplot(424)
    plt.scatter(X_eff_sta2[:,7],X_eff_sta2[:,1], c='r', marker='x')
    plt.scatter(X_som_sta2[:,7],X_som_sta2[:,1], c='w', marker='o', s=5)
    plt.xlabel('maxA_mean')
    plt.ylabel('Dur')
    plt.subplot(426)
    plt.scatter(X_eff_sta2[:,7],X_eff_sta2[:,2], c='r', marker='x')
    plt.scatter(X_som_sta2[:,7],X_som_sta2[:,2], c='w', marker='o', s=5)
    plt.xlabel('maxA_mean')
    plt.ylabel('cent_f')
    plt.subplot(428)
    plt.scatter(X_eff_sta2[:,7],X_eff_sta2[:,3], c='r', marker='x')
    plt.scatter(X_som_sta2[:,7],X_som_sta2[:,3], c='w', marker='o', s=5)
    plt.xlabel('maxA_mean')
    plt.ylabel('A')
    
    
    fig.savefig('maxA_mean1.png')
    
    fig, axes= plt.subplots(4, 2)
    
    plt.subplot(421)
    plt.scatter(X_eff_sta1[:,7],X_eff_sta1[:,4], c='r', marker='x') #K_Dur eff
    plt.scatter(X_som_sta1[:,7],X_som_sta1[:,4], c='w', marker='o', s=5) #K_Dur som
    plt.xlabel('maxA_mean')
    plt.ylabel('Dur/A')
    plt.title('sta1')
    plt.subplot(423)
    plt.scatter(X_eff_sta1[:,7],X_eff_sta1[:,5], c='r', marker='x') #K_cent_f
    plt.scatter(X_som_sta1[:,7],X_som_sta1[:,5], c='w', marker='o', s=5)
    plt.xlabel('maxA_mean')
    plt.ylabel('dom_f')
    plt.subplot(425)
    plt.scatter(X_eff_sta1[:,7],X_eff_sta1[:,6], c='r', marker='x') #K_A
    plt.scatter(X_som_sta1[:,7],X_som_sta1[:,6], c='w', marker='o', s=5)
    plt.xlabel('maxA_mean')
    plt.ylabel('E')
    plt.subplot(427)
    #plt.scatter(X_eff_sta1[:,7],X_eff_sta1[:,8], c='r', marker='x') #K_Dur/A
    #plt.scatter(X_som_sta1[:,7],X_som_sta1[:,8], c='w', marker='o', s=5)
    plt.xlabel('maxA_mean')
    plt.ylabel('AsDec')
    
    #station 2 colonne 2
    plt.subplot(422)
    plt.scatter(X_eff_sta2[:,7],X_eff_sta2[:,4], c='r', marker='x')
    plt.scatter(X_som_sta2[:,7],X_som_sta2[:,4], c='w', marker='o', s=5)
    plt.xlabel('maxA_mean')
    plt.ylabel('Dur/A')
    plt.title('sta2')
    plt.subplot(424)
    plt.scatter(X_eff_sta2[:,7],X_eff_sta2[:,5], c='r', marker='x')
    plt.scatter(X_som_sta2[:,7],X_som_sta2[:,5], c='w', marker='o', s=5)
    plt.xlabel('maxA_mean')
    plt.ylabel('dom_f')
    plt.subplot(426)
    plt.scatter(X_eff_sta2[:,7],X_eff_sta2[:,6], c='r', marker='x')
    plt.scatter(X_som_sta2[:,7],X_som_sta2[:,6], c='w', marker='o', s=5)
    plt.xlabel('maxA_mean')
    plt.ylabel('E')
    plt.subplot(428)
    #plt.scatter(X_eff_sta2[:,7],X_eff_sta2[:,8], c='r', marker='x')
    #plt.scatter(X_som_sta2[:,7],X_som_sta2[:,8], c='w', marker='o', s=5)
    plt.xlabel('maxA_mean')
    plt.ylabel('AsDec')
    
    fig.savefig('maxA_mean2.png')

