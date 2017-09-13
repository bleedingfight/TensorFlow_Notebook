import numpy as np
import os 
import utils.utilities import *
from sklearn.model_selection import train_test.split
import matplotlib.pyplot as plt
X_train,labels_train,list_ch_train=read_data(data_path="./data/",split=train)
X_test,labels_test,list_ch_test = read_data(data_path='./data/',split='train')
assert list_cn_train == list_ch_test,"Mismatch in channels!"
X_train,X_test = standardize(X_train,X_test)

X_tr, X_vld, lab_tr, lab_vld = train_test_split(X_train, labels_train,                            stratify = labels_train, random_state = 123)
y_tr = one_hot(lab_tr)
y_vld = one_hot(lab_vld)
y_test = one_hot(labels_test)
