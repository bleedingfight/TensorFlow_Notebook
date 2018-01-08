#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 25 15:14:21 2017

@author: liushuai
"""

from sklearn.learning_curve import validation_curve
from sklearn.datasets import load_digits
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import numpy as np
param_range = np.logspace(-6,-2.3,10)
digits = load_digits()
X_data = digits.data
y_data = digits.target
train_loss,test_loss = validation_curve(
        SVC(),X_data,y_data,param_name ='gamma',param_range= param_range,cv=10,scoring='neg_mean_absolute_error',
        )
train_loss_mean = -np.mean(train_loss,axis=1)
test_loss_mean = -np.mean(test_loss,axis=1)
plt.plot(param_range,train_loss_mean,'o-',color='r')
plt.plot(param_range,test_loss_mean,'o-',color='b')
plt.xlabel('Gamma')
plt.legend(loc='best')