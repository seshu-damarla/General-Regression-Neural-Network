# -*- coding: utf-8 -*-
"""
Created on Fri Jan 28 12:52:44 2022

@author: Seshu Kumar Damarla
"""

"""
Python program for General Regression Neural Network

"""

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt


# data
xtrain = pd.read_csv('trainxdata.csv',header=None)
ytrain = pd.read_csv('trainydata.csv', header=None)

xtrain = np.array(xtrain)
ytrain = np.array(ytrain)

xtest = pd.read_csv('testxdata.csv', header=None)
ytest = pd.read_csv('testydata.csv', header=None)

xtest = np.array(xtest)
ytest = np.array(ytest)

# normaliation
xmean = np.mean(xtrain, axis=0, keepdims=True, dtype=np.float)
xstd = np.std(xtrain, axis=0, keepdims=True, dtype=np.float)

ymean = np.mean(ytrain, axis=0, keepdims=True, dtype=np.float)
ystd = np.std(ytrain, axis=0, keepdims=True, dtype=np.float)

xtrain = (xtrain-xmean)/xstd
ytrain = (ytrain-ymean)/ystd

# output of hidden layer
nex=xtrain.shape[0]
ni=xtrain.shape[1]

sigma=0.11;           # spread of RBF neurons in the pattern layer
centers=xtrain       # centers of pattern layer units 
A=ytrain             # weights of summation layer unit 1
B=np.ones((nex,1))   # weights of summation layer unit 2
ypred=np.zeros((xtest.shape[0],1))
xtest = (xtest-xmean)/xstd

for i in range(0, xtest.shape[0], 1):
#    H=np.zeros((nex,1))
    sample=xtest[i,:]
    h=np.linalg.norm((sample-centers), ord=None, axis=1, keepdims=True)
    h=np.exp(-(h**2)/(2*sigma*sigma))
    uA=np.sum(h*A)
    uB=np.sum(h*B)
    ypred[i,:]=uA/uB
#    print(ybar)
#    ypred.append(ybar)
    
ypred = ypred*ystd+ymean
#print(ypred)
plt.plot(ypred, label='GRNN')
plt.plot(ytest, label='Actual Data')
plt.legend()

(R, pval) = stats.pearsonr(ytest.flatten(),ypred.flatten())
print(R)
        
        
    
    
    
        




