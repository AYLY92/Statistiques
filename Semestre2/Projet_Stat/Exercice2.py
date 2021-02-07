# -*- coding: utf-8 -*-
"""
Created on Mon Feb  1 20:39:58 2021

@author: almamy
"""

# Scientific and vector computation for python
import numpy as np
# Plotting library
from matplotlib import pyplot as plt
from sklearn import linear_model
from sklearn import datasets
from sklearn.linear_model import RidgeCV, LassoCV

#################################QUESTION_1################################################

#1. Programmation de la fonction regression(X,Y)
def regression(X, Y):
    return np.linalg.pinv(np.transpose(X).dot(X)).dot(np.transpose(X).dot(Y))


#Appliquons la fonction sur le jeu de données Boston House prices
#chargement et visualisation des données
boston = datasets.load_boston()
print(boston)
print(boston.data)
print(boston.feature_names)
print(boston.target)

#Vérification des dimensions
boston.data.shape
boston.target.shape
#Redimensionnement de boston.target
boston.target = boston.target.reshape(boston.target.shape[0],1)

#Création des matrices X et Y
X = np.insert(np.ones((boston.data.shape[0], 1)), [-1], boston.data, axis = 1)
Y = boston.target

#Application de la fonction regression(X,Y)
regression(X,Y)
alpha = regression(X,Y)[0:-1]
beta = regression(X, Y)[-1]
#Comparons les vecteurs obtenus avec les attributs coef_ et intercept_
from sklearn.linear_model import LinearRegression
# Create a Linear regressor
lm = LinearRegression().fit(X, Y)
lm.intercept_
lm.coef_

"""On peut dire que alpha et beta sont respectivement égaux à coef_ et intercept_"""

###################################QUESTION_2##############################################

#Fonction regress
def regress(X, alpha, beta):
    return  np.dot(X,alpha) + beta
    
    
###################################QUESTION_3##############################################

#variable cible Y
Y = boston.target
#valeur prédite
Yp = regress(X,np.transpose(lm.coef_), lm.intercept_)
#calcul de l'erreur
def erreur(Y, Yp):
    print("L'erreur  est égale à:", np.sum((Y - Yp)**2))
erreur(Y, Yp)

###################################QUESTION_4##############################################

#a. Programmons la régression ridge
I = np.identity(X.shape[1])
def ridge_regression(X, Y, lambda_):
    return np.dot (np.dot (np.linalg.inv (np.dot (X.T, X) + lambda_*I), X.T), Y)

#Comparons les vecteurs obtenus  sur le jeu de données Boston avec coef_ et intercept_
#initialisation de lamba_
lambda_ = 1
#ridge fonction
ridge_regression(X, Y, lambda_)
#valeur de alpha et beta
alpha = ridge_regression(X, Y, lambda_)[0:-1]
beta = ridge_regression(X, Y, lambda_)[-1]
#Implémentatin et entrainement du modéle
from sklearn.linear_model import Ridge
rr = Ridge(alpha = 1)
rr.fit(X, Y)
rr.coef_
rr.intercept_
"""Les valeurs de alpha et beta sont différentes de rr.coef_ et rr.intercept_"""

#b. Evolution des coéfficients du vecteur alpha_ en fonction du param de régularisation
n_alphas = 200
alphas = np.logspace(-2, 3, n_alphas)
    
coefs = []
for i in alphas:  
    ridge = linear_model.Ridge(alpha = i, fit_intercept=False)
    ridge.fit(X, Y)
    coefs.append(ridge.coef_)
    
coefs = np.array(coefs) 
coefs = coefs.reshape(coefs.shape[0], -1)

fig = plt.figure(1, figsize=(10, 10))

ax = plt.gca()
ax.plot(alphas, coefs)
ax.set_xscale('log')
#ax.set_xlim(ax.get_xlim()[::-1])  # reverse axis
plt.xlabel('lambda')
plt.ylabel('coefficients')
plt.title('Evolution des coefficients du vecteur âlpha en fonction du paramétre de régularisation')
plt.axis('tight')
plt.legend(labels = ['CRIM', 'ZN', 'INDUS',  'CHAS',  'NOX','RM', 'AGE', 'DIS', 'RAD' ,
                     'TAX', 'PTRATIO', 'B' ,'LSTAT' , 'PRICE'])
plt.show()

"""D'aprés le graphe, les variables les plus influentes sont: CHAS, DIS, B, 
ZN et RM"""

#c. Meilleure valeur pour le paramétre lambda
regr_cv = RidgeCV(alphas )
# Entrainement sur les données Boston
model_cv = regr_cv.fit(X, Y) 
# Valeur de lambda
model_cv.alpha_

#Calcul de l'erreur
#Vecter des étiquettes prédites:Yp
Yp = model_cv.predict(X)
#Valeur de l'erreur 
ridge_error = erreur(Y, Yp)


###################################QUESTION_5##############################################

#a. Tracez l’évolution des coefficients du vecteur âlpha en f(lambda).

n_alphas = 2000
alphas = np.logspace(-2, 3, n_alphas)

    
coefs_lasso = []
for i in alphas:  
    lasso = linear_model.Lasso(alpha=i, fit_intercept=False)
    lasso.fit(X, Y)
    coefs_lasso.append(lasso.coef_)
    
coefs_lasso = np.array(coefs_lasso) 
coefs_lasso = coefs_lasso.reshape(coefs_lasso.shape[0], -1)

fig = plt.figure(1, figsize=(10, 10))

ax = plt.gca()
ax.plot(alphas, coefs_lasso)
ax.set_xscale('log')
#ax.set_xlim(ax.get_xlim()[::-1])  # reverse axis
plt.xlabel('lambda')
plt.ylabel('coefficients')
plt.title('Evolution des coefficients du vecteur âlpha en fonction du paramétre de régularisation')
plt.axis('tight')
plt.legend(labels = ['CRIM', 'ZN', 'INDUS',  'CHAS',  'NOX','RM', 
                     'AGE', 'DIS', 'RAD' ,'TAX',  'PTRATIO', 'B' ,'LSTAT' , 'PRICE'])
plt.show()

"""D'aprés le graphe, les variables plus influentes sont: CHAS,
DIS, B, ZN et RM.
Oui, elles sont les mêmes que celles trouvées au niveau de la question précédente.
Les autres variables restent constantes lorsque la valeur de lambda augmente."""

#b. Meilleure valeur pour le paramétre lambda

regr_lasso = LassoCV(cv = 200 )
# Entrainement sur les données Boston
model_lasso = regr_lasso.fit(X, Y) 
# Valeur de lambda
model_lasso.alpha_

#Calcul de l'erreur 
Yp = model_lasso.predict(X)
lasso_error = erreur(Y, Yp)

