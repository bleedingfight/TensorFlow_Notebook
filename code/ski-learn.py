#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 25 08:20:03 2017

@author: liushuai
"""
import matplotlib.pyplot as plt
from sklearn import datasets
from  sklearn.linear_model import LinearRegression
loaded_data = datasets.load_boston()
data_X = loaded_data.data
data_y = loaded_data.target
model = LinearRegression()
model.fit(data_X,data_y)
print(model.predict(data_X[:4,:]))
print(data_y[:4])
x,y = datasets.make_regression(n_samples=100,n_features=1,n_targets=1,noise=4)
plt.plot(x,y,'o')