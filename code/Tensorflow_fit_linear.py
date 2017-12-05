# -*- coding: utf-8 -*-
"""
Created on Tue Mar  7 19:35:11 2017

@author: Alien
"""

import numpy as np
import tensorflow as tf
with tf.name_scope('Input_data'):
    x_data = np.random.rand(100).astype(np.float32)
    y_data = x_data*0.1 + 0.3
with tf.name_scope('weight'):
    Weights = tf.Variable(tf.random_uniform([1],-1,1))
    tf.summary.histogram('weights',Weights)
with tf.name_scope('Bias'):
    Bias = tf.Variable(tf.zeros([1]))
    tf.summary.histogram('Bias',Bias)
y = Weights*x_data + Bias
with tf.name_scope('loss'):
    loss = tf.reduce_mean(tf.square(y-y_data))
    tf.summary.scalar('loss',loss)
optimizer = tf.train.GradientDescentOptimizer(0.5)
with tf.name_scope('train'):
    train = optimizer.minimize(loss)
init = tf.global_variables_initializer()
sess = tf.Session()
merged = tf.summary.merge_all()
writer = tf.summary.FileWriter('logs/',sess.graph)
sess.run(init)
for i in range(201):
    sess.run(train)
    if i%20 ==0:
        result = sess.run(merged)
        writer.add_summary(result,i)
        print("第%d次迭代结果，Weight%.4f,Bias%.4f"%(i,sess.run(Weights),sess.run(Bias)))
        
