#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 25 08:31:07 2017

@author: liushuai
"""

from sklearn import datasets
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
load_data = datasets.load_boston()
data_X = load_data.data
data_y = load_data.target
model = LinearRegression()
model.fit(data_X,data_y)
print(model.coef_)
print(model.intercept_)
print(model.get_params())
print(model.score(data_X,data_y))#R^2 coefficient of determination 

