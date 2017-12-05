# -*- coding: utf-8 -*-
"""
Created on Fri Mar  3 16:39:54 2017

@author: Alien
"""
import tensorflow as tf
import input_data
mnist = input_data.read_data_sets("MNIST_data",one_hot = True)
sess = tf.InteractiveSession()
x = tf.placeholder("float",shape=[None,784])
y_ = tf.placeholder("float",shape=[None,10])
W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))
sess.run(tf.global_variables_initializer())
y = tf.nn.softmax(tf.matmul(x,W)+b)
cross_entropy = -tf.reduce_sum(y_*tf.log(y))
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)
for i in range(1000):
    batch = mnist.train.next_batch(50)
    train_step.run(feed_dict={x:batch[0],y_:batch[1]})
correct_prediction = tf.equal(tf.arg_max(y,1),tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction,"float"))
print("预测精度:",accuracy.eval(feed_dict={x:mnist.test.images,y_:mnist.test.labels}))