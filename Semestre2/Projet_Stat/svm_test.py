# -*- coding: utf-8 -*-
"""
Created on Wed Jan 13 02:03:42 2021

@author: lucho
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
import sklearn.metrics as sm


df = pd.read_csv("data_svm.csv",sep=";")
df = df.astype({'x1':'float64', 'x2':'float64', 'y':'category'})

#fonction pour graphique nuage de points
#entrées : data.frame avec l’ensemble des individus
# data.frame relatifs aux individus positifs et négatifs
def myscatter(df,dfpos,dfneg):
    #nuage de points « blanc » pour définir les dimensions du graphique
    plt.scatter(df.iloc[:,0],df.iloc[:,1],color="white")
    #annotate - positive instances
    for i in dfpos.index:
        plt.annotate(i,xy=(df.loc[i,'x1'],df.loc[i,'x2']),xytext=(-3,-3),textcoords='offset points',color='red')

     #annotate - negative instances
    for i in dfneg.index:
        plt.annotate(i,xy=(df.loc[i,'x1'],df.loc[i,'x2']),xytext=(-3,-3),textcoords='offset points',color='blue')
    return None


myscatter(df, df[df.y==-1],df[df.y==1])

svm = SVC(kernel='linear')
svm.fit(df.to_numpy()[:,0:2],df.to_numpy()[:,2].astype('int'))
svm.support_vectors_
svm.support_
svm.dual_coef_

myscatter(df, df[df.y==-1],df[df.y==1])
c1 = svm.support_vectors_[:,0]
c2 = svm.support_vectors_[:,1]
plt.scatter(c1,c2,s=200,facecolors='none',edgecolors='black')
plt.show()

#w(w1, w2)
svm.coef_
#b
svm.intercept_


## Représentation des frontières 
#calcul des coordonnés de deux points qui passent par la droite 

#frontiere f(x) = W'x+b = 1
xh = np.array([2,11])

yh = -svm.coef_[0][0]/svm.coef_[0][1]*xh-(svm.intercept_+1.0)/svm.coef_[0][1]
#frontiere f(x) = W'x+b = -1
xb = np.array([4.5,12])
yb = -svm.coef_[0][0]/svm.coef_[0][1]*xb-(svm.intercept_-1.0)/svm.coef_[0][1]

#frontiere f(x) = W'x+b = 0
xf = np.array([3,12])
yf = -svm.coef_[0][0]/svm.coef_[0][1]*xf-svm.intercept_/svm.coef_[0][1]


myscatter(df, df[df.y==-1],df[df.y==1])
c1 = svm.support_vectors_[:,0]
c2 = svm.support_vectors_[:,1]
plt.scatter(c1,c2,s=200,facecolors='none',edgecolors='black')
plt.plot(xf,yf,c='green')
plt.plot(xb,yb,c='gray')
plt.plot(xh,yh,c='gray')
plt.show()


#Evaluation
svm.score(df.to_numpy()[:,0:2],df.to_numpy()[:,2].astype('int'))
svm.predict(df.to_numpy()[:,0:2].astype('int'))
sm.confusion_matrix(df.to_numpy()[:,2].astype('int'), svm.predict(df.to_numpy()[:,0:2]))
sm.plot_confusion_matrix(svm,df.to_numpy()[:,0:2],df.to_numpy()[:,2].astype('int'))  
