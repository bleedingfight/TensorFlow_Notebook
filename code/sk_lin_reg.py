#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 18 16:23:29 2017

@author: liushuai
"""

from sklearn import linear_model
reg = linear_model.LinearRegression()
reg.fit([[18,26.86],[20,1],[2,2]],[0,1,2])
print(reg.coef_)