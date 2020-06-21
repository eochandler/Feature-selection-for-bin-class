#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 22 14:04:10 2020

@author: elijahchandler
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import datasets

#question 1:
    #700 is the best values of lmbda
#question 2:
    #sex, limit balance Pay_0, Pay_2, Pay_3 are the most important features
    


info = pd.read_csv('/Users/elijahchandler/Downloads/UCI_Credit_Card.csv')

Y = info['default.payment.next.month']
X = info.drop('default.payment.next.month', axis = 1)

#%%
X_train, X_test, Y_train, Y_test = train_test_split(X,Y)
#%%
X_train = X_train.values
Y_train = Y_train.values
X_train = (X_train - X_train.mean(axis = 0))/X_train.std(axis = 0)
#%%

def clip(beta, alpha):
    clipped = np.minimum(beta, alpha)
    clipped = np.maximum(clipped, -alpha)
    return clipped


def sigmoid(u):
    return np.exp(u)/(1+ np.exp(u))


def crossEntropy(p,q):
    return -p*np.log(q) - (1-p) * np.log(1-q)

def h(u, yi):
    exp = np.exp(u)
    return -yi * u + np.log(1 + exp)

def L(beta, X, Y):
    N = X.shape[0]
    mySumHi = 0
    for i in range(N):
        xihat = X[i]
        yi = Y[i]
        dotProduct = np.vdot(xihat, beta)
        mySumHi += h(dotProduct, yi)
    return mySumHi



#%%
beta = np.random.randn(20)
alpha = .5
clipped = clip(beta, alpha)

plt.plot(beta)
plt.plot(clipped)

def proxL1Norm(betaHat, alpha ,penalizeAll = True):
    out = betaHat - clip(betaHat, alpha)
    
    if not penalizeAll:
        out[0] = betaHat[0]
        
    return out
#%%

def logReg_L1Reg_ProxGrad(X,Y,lmbda):
#gradient descent
    
    maxIter = 75
    alpha = .00001
    N,d = X.shape
    #k = Y.shape[1]
    
    allOnes = np.ones((N, 1))
    X = np.hstack((allOnes, X))
    
    beta = np.zeros((d+1))
    gradNorms = []
    costVals = []
    for idx in range(maxIter):
        
        grad = np.zeros(d+1)
        
        for i in range(N):
            Xi = X[i, :]
            Yi = Y[i]
            qi = sigmoid(np.vdot(Xi, beta))
            
            grad += (qi - Yi) * Xi
        
    #recompute beta
    #now we're using the proximal gradient methd
        beta = proxL1Norm(beta-alpha*grad, alpha*lmbda)
    
    #find the norm 
        nrm = np.linalg.norm(grad)
        gradNorms.append(nrm)
        
        #computerd w/ L1 Reg
        cost = L(beta,X,Y) + lmbda*np.sum(np.abs(beta))
        costVals.append(cost)
        print(idx, cost)
        
    return beta, gradNorms, costVals 

lmbda = 700
xVals = np.arange(-np.pi, np.pi, .01)
betaHat = np.sin(xVals)
prox = proxL1Norm(betaHat, .2)
plt.plot(xVals, betaHat)
plt.plot(xVals, prox)
beta,gradnorms, costVals = logReg_L1Reg_ProxGrad(X_train, Y_train, lmbda);


#l1 encourages beta to be sparse the non zero components of beta corespond to the most important features
#first non zeros corespond to 


np.nonzero(beta)
#%%

