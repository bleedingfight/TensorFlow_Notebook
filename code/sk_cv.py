#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 25 09:39:34 2017

@author: liushuai
"""

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import matplotlib.pyplot as plt
iris = load_iris()
X_data = iris.data
y_data = iris.target
X_train,X_test,y_train,y_test = train_test_split(X_data,y_data,test_size=0.3,random_state=22)
model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_train,y_train)
print('Random Cross_validation:%.4f'%model.score(X_test,y_test))
score = cross_val_score(model,X_data,y_data,cv=5,scoring='accuracy')#score used in classfication
print('Cross_validation The Whole Data:%.4f'%score.mean())
# Test n_neightbors = 5
scores = []
k_neigh = []
for i in range(1,31):
    mod = KNeighborsClassifier(n_neighbors=i)
#    scor = cross_val_score(mod,X_data,y_data,cv=5,scoring='accuracy')
    scor = cross_val_score(mod,X_data,y_data,cv=5,scoring='neg_mean_absolute_error')
    m = scor.mean()
    scores.append(m)
    k_neigh.append(i)
    print('n_neighbors=%d,score:%.4f'%(i,m))
max_score = max(scores)
index = np.argmax(scores)
print('max value is :%.4f'%max_score)
print('The Index Of The Max Value is %d'%index)
plt.plot(k_neigh,scores,'r',lw=2)
plt.grid(True)
plt.hold(True)
plt.plot(k_neigh[index],max_score,'bo')