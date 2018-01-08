#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 25 15:30:01 2017

@author: liushuai
"""

from sklearn import svm
from sklearn import datasets
clf = svm.SVC()
iris = datasets.load_iris()
X_data,y_data = iris.data,iris.target
import pickle
with open('save/clf.pickle','wb') as f:
    pickle.dump(clf,f)