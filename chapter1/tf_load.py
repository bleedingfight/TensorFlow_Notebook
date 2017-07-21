import tensorflow as tf
import numpy as np
import os
from tensorflow.examples.tutorials.mnist import input_data
def init_weights(shape):
    return tf.Variable(tf.random_normal(shape,mean=0,stddev=0.01))
def model(X,w_h,w_h2,w_o,p_keep_input,p_keep_hidden):
    X = tf.nn.dropout(X,p_keep_input)
    h = tf.nn.relu(tf.matmul(X,w_h))
    h = tf.nn.dropout(w_h2,p_keep_hidden)
    h2 = tf.nn.relu(tf.matmul(h,w_h2))
    h2 = tf.nn.dropout(h2,p_keep_hidden)
    return tf.matmul(w_h2,w_o)
global_step = tf.Variable(0,name='global_step',trainable=False)
mnist = input_data.read_data_sets("/home/hpc/文档/mnist_tutorial/mnist")
trX,trY,teX,teY = mnist.train.images,mnist.train.labels,mnist.test.images,mnist.test.labels
X = tf.placeholder("float",[None,784])
Y = tf.placeholder("float",[None,10])
w_h = init_weights([784,625])
w_h2 = init_weights([625,625])
w_o = init_weights([625,10])
p_keep_input = tf.placeholder("float")
p_keep_hidden = tf.placeholder("float")
ckpt_dir = "/home/hpc/文档/Tensorflow/checkpoint_file/"
if not os.path.exists(ckpt_dir):
    os.makedirs(ckpt_dir)
global_step = tf.Variable(0,name='global_step',trainable=False)
saver = tf.train.Saver()
py_x = model(X,w_h,w_h2,w_o,p_keep_input,p_keep_hidden)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y,logits=py_x))
train_op = tf.train.RMSPropOptimizer(0.001,0.9).minimize(cost)
predict_op = tf.argmax(py_x,1)
non_storable_variable = tf.Variable(777)
with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    start = global_step.eval()
    for i in range(start,128):
        for start,end in zip(range(0,len(trX),128),range(128,len(trX)+1,128)):
            sess.run(train_op,feed_dict={X:trX[start:end],Y:trY[start:end],p_keep_input:0.8,p_keep_hidden:0.5})
        global_step.assign(i).eval()
        saver.save(sess,ckpt_dir+"mnist_2017.ckpt",global_step=global_step)
        
