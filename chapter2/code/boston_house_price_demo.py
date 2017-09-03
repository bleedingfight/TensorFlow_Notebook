"""
Created on Mon Sep  4 00:09:03 2017
Win10 64bit TensorFlow1.3
@author: Alien
"""

import itertools
import pandas as pd 
import tensorflow as tf 
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2' 
tf.logging.set_verbosity(tf.logging.INFO)
train_url = 'http://download.tensorflow.org/data/boston_train.csv'
test_url = 'http://download.tensorflow.org/data/boston_test.csv'
prediction_url = 'http://download.tensorflow.org/data/boston_predict.csv'
columns = ['cris','zn','indus','nox','rm','age','dis','tax','ptratio','medv']
feature = ['cris','zn','indus','nox','rm','age','dis','tax','ptratio']
label = 'medv'
training_set = pd.read_csv(train_url,skipinitialspace=True,skiprows=1,names=columns)
testing_set = pd.read_csv(test_url,skipinitialspace=True,skiprows=1,names=columns)
prediction_set = pd.read_csv(prediction_url,skipinitialspace=True,skiprows=1,names=columns)
feature_cols = [tf.feature_column.numeric_column(k) for k in feature]
regressor = tf.estimator.DNNRegressor(feature_columns=feature_cols,hidden_units=[10,10],model_dir='./')
def get_input_fn(data_set,num_epochs=None,shuffle=True):
	return tf.estimator.inputs.pandas_input_fn(
		x = pd.DataFrame({k:data_set[k].values for k in feature}),
		y = pd.Series(data_set[label].values),
		num_epochs = num_epochs,
		shuffle = shuffle)
regressor.train(input_fn=get_input_fn(training_set),steps=5000)