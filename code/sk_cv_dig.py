#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 25 11:26:50 2017

@author: liushuai
"""

from sklearn.learning_curve import learning_curve
from sklearn.datasets import load_digits
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
digit = load_digits()
X_data = digit.data
y_data = digit.target
train_sizes,train_loss,test_loss=learning_curve(
        SVC(gamma=0.001),X_data,y_data,cv=10,scoring='neg_mean_absolute_error',
        train_sizes=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1])
train_loss_mean = -np.mean(train_loss,axis=1)
test_loss_mean = -np.mean(test_loss,axis=1)
plt.plot(train_sizes,train_loss_mean,'o-',color="r",label='Training')
plt.plot(train_sizes,test_loss_mean,'o-',color="b",label='Testing')
plt.legend()