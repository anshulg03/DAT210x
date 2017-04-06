import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import numpy as np
from sklearn import preprocessing
from sklearn import manifold
from sklearn.decomposition import PCA

#Loading Data

x = pd.read_csv('C:/Users/anshangu/Documents/GitHub/DAT210x/Module6/Datasets/parkinsons.data')

x = x.drop('name', axis = 1)

y = x.status

x.drop('status', axis = 1, inplace = True)


for i in range(2,5):
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = 7)
    
    norm = preprocessing.Normalizer()
    MAS = preprocessing.MaxAbsScaler()
    MMS = preprocessing.MinMaxScaler()
    KC = preprocessing.KernelCenterer()
    SS = preprocessing.StandardScaler()
    
    #norm.fit(X_train)
    #X_train = norm.transform(X_train)
    #X_test = norm.transform(X_test)
    
    #MAS.fit(X_train)
    #X_train = MAS.transform(X_train)
    #X_test = MAS.transform(X_test)
    #
    #MMS.fit(X_train)
    #X_train = MMS.transform(X_train)
    #X_test = MMS.transform(X_test)
    
    #KC.fit(X_train)
    #X_train = KC.transform(X_train)
    #X_test = KC.transform(X_test)
    
    SS.fit(X_train)
    X_train = SS.transform(X_train)
    X_test = SS.transform(X_test)
    
#    pca = PCA(n_components=i)
#    X_train = pca.fit_transform(X_train)
#    X_test = pca.transform(X_test)
    
    iso = manifold.Isomap(n_neighbors = i ,n_components = 6)
    X_train = iso.fit_transform(X_train)
    X_test = iso.transform(X_test)
    
    
    c_best = 0
    g_best = 0
    best_score = 0
    for C in np.arange(0.05, 2 , 0.05):
        for g in np.arange(0.001, 0.1, 0.001):
            model = SVC(C=C, gamma = g)
            model.fit(X_train, y_train)
            score = model.score(X_test, y_test)
            if best_score < score:
                best_score = score
                c_best = C
                g_best = g
                
    
    print(best_score, '   ', g, '    ', C)
        