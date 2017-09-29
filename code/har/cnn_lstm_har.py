import numpy as np
import os
from utilities import *
from sklearn.model_selection import train_test_split
import tensorflow as tf
import matplotlib.pyplot as plt
data_path = './UCI HAR Dataset'
X_train,labels_train,list_ch_train = read_data(data_path=data_path,split='train')
X_test,labels_test,list_ch_test = read_data(data_path=data_path,split='test')
assert list_ch_train == list_ch_test, "Mistmatch in channels!"
X_tr,X_vld,lab_tr,lab_vld = train_test_split(X_train,labels_train,test_size = 0.2,
        stratify = labels_train,random_state =123)
y_tr = one_hot(lab_tr)
y_vld = one_hot(lab_vld)
y_test = one_hot(labels_test)
lstm_size = 36
lstm_layers = 2
batch_size = 600
seq_len = 128
learning_rate = 0.0005
n_classes = 6
n_channels = 9
epochs = 500

graph = tf.Graph()
with graph.as_default():
    inputs_ = tf.placeholder(tf.float32,[None,seq_len,n_channels],name='inputs')
    labels_ = tf.placeholder(tf.float32,[None,n_classes],name = 'labels')
    keep_prob_ = tf.placeholder(tf.float32,name = 'keep')
    learning_rate_ = tf.placeholder(tf.float32,name = 'learning_rate')
with graph.as_default():
    conv1 = tf.layers.conv1d(inputs = inputs_,filters=18,kernel_size=2,strides=1,
            padding='same',activation = tf.nn.relu)
    n_ch = n_channels*2
with graph.as_default():
    lstm_in = tf.transpose(conv1,[1,0,2])
    lstm_in = tf.reshape(lstm_in,[-1,n_ch])
    lstm_in = tf.layers.dense(lstm_in,lstm_size,activation = tf.nn.relu)
    lstm_in = tf.split(lstm_in,seq_len,0)
    lstm = tf.contrib.rnn.BasicLSTMCell(lstm_size)
    drop = tf.contrib.rnn.DropoutWrapper(lstm,output_keep_prob = keep_prob_)
    cell = tf.contrib.rnn.MultiRNNCell([drop]*lstm_layers)
    initial_state = cell.zero_state(batch_size,tf.float32)

with graph.as_default():
    outputs,final_state = tf.contrib.rnn.static_rnn(cell,lstm_in,dtype = tf.float32,initial_state = initial_state)
    logits = tf.layers.dense(outputs[-1],n_classes,name='logits')
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits,labels=labels_))
    train_op = tf.train.AdamOptimizer(learning_rate_)
    gradients = train_op.compute_gradients(cost)
    capped_gradients = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gradients]
    optimizer = train_op.apply_gradients(capped_gradients)
    correct_pred = tf.equal(tf.argmax(logits,1),tf.argmax(labels_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred,tf.float32),name = 'accuracy')

if (os.path.exists('checkpoints-crnn') == False):
    os.mkdir('checkpoints-crnn')
validation_acc = []
validation_loss = []
train_acc = []
train_loss = []
with graph.as_default():
    saver = tf.train.Saver()
#with tf.Session(graph=graph) as sess:
#    sess.run(tf.global_variables_initializer())
#    iteration = 1
#    for e in range(epochs):
#        state = sess.run(initial_state)
#        for x,y in get_batches(X_train,y_tr,batch_size):
#            feed = {inputs_:x,labels_:y,keep_prob_:0.5,
#                    initial_state:state,learning_rate_:learning_rate}
#            loss,_,state,acc = sess.run([cost,optimizer,final_state,
#                accuracy],feed_dict = feed)
#            train_acc.append(acc)
#            train_loss.append(loss)
#            if(iteration%5 == 0):
#                print('Epoch:{}/{}'.format(e,epochs),
#                        'Iteration:{:d}'.format(iteration),
#                        'Train loss:{:6f}'.format(loss),
#                        'Train acc:{:.6f}'.format(acc))
#            if(iteration%25 == 0):
#                val_state = sess.run(cell.zeros_state(batch_size,tf.float32))
#                val_acc_ = []
#                val_loss_ = []
#                for x_v,y_v in get_batches(X_vld,y_vld,batch_size):
#                    feed = {input_:x_v,labels_:y_v,keep_prb_:1.,
#                            initial_state:val_state}
#                    val_acc.append(acc_v)
#                    val_loss_.append(loss_v)
#                    print('Epoch:{}/{}'.format(e,epochs),
#                            "Iteration:{:d}".format(iteration),
#                            'Validation loss:{:6f}'.format(np.mean(val_loss_)),
#                            'Validation acc:{:.6f}'.format(np.mean(val_acc_)))
#                    validation_acc.append(np.mean(val_acc_))
#                    validation_loss.append(np.mean(val_loss_))
#            iteration+=1
#
#    saver.save(sess,"checkpoints-crnn/har.ckpt")
with tf.Session(graph=graph) as sess:
    sess.run(tf.global_variables_initializer())
    iteration = 1
    for e in range(epochs):
        # Initialize 
        state = sess.run(initial_state)
        # Loop over batches
        for x,y in get_batches(X_tr, y_tr, batch_size):
            # Feed dictionary
            feed = {inputs_ : x, labels_ : y, keep_prob_ : 0.5,
                    initial_state : state, learning_rate_ : learning_rate}
            loss, _ , state, acc = sess.run([cost, optimizer, final_state, accuracy], feed_dict = feed)
            train_acc.append(acc)
            train_loss.append(loss)
            # Print at each 5 iters
            if (iteration % 5 == 0):
                print("Epoch: {}/{}".format(e, epochs),
                      "Iteration: {:d}".format(iteration),
                      "Train loss: {:6f}".format(loss),
                      "Train acc: {:.6f}".format(acc))
            # Compute validation loss at every 25 iterations
            if (iteration%25 == 0):
                # Initiate for validation set
                val_state = sess.run(cell.zero_state(batch_size, tf.float32))
                val_acc_ = []
                val_loss_ = []
                for x_v, y_v in get_batches(X_vld, y_vld, batch_size):
                    # Feed
                    feed = {inputs_ : x_v, labels_ : y_v, keep_prob_ : 1.0, initial_state : val_state}
                    # Loss
                    loss_v, state_v, acc_v = sess.run([cost, final_state, accuracy], feed_dict = feed)
                    val_acc_.append(acc_v)
                    val_loss_.append(loss_v)
                # Print info
                print("Epoch: {}/{}".format(e, epochs),
                      "Iteration: {:d}".format(iteration),
                      "Validation loss: {:6f}".format(np.mean(val_loss_)),
                      "Validation acc: {:.6f}".format(np.mean(val_acc_)))
                # Store
                validation_acc.append(np.mean(val_acc_))
                validation_loss.append(np.mean(val_loss_))
            iteration += 1
    saver.save(sess,"checkpoints-crnn/har.ckpt")
with tf.Session(graph=graph) as sess:
    sess.run(tf.global_variables_initializer())
    iteration = 1
    for e in range(epochs):
        # Initialize 
        state = sess.run(initial_state)
        # Loop over batches
        for x,y in get_batches(X_tr, y_tr, batch_size):
            # Feed dictionary
            feed = {inputs_ : x, labels_ : y, keep_prob_ : 0.5,
                    initial_state : state, learning_rate_ : learning_rate}
            loss, _ , state, acc = sess.run([cost, optimizer, final_state, accuracy], feed_dict = feed)
            train_acc.append(acc)
            train_loss.append(loss)
            # Print at each 5 iters
            if (iteration % 5 == 0):
                print("Epoch: {}/{}".format(e, epochs),
                      "Iteration: {:d}".format(iteration),
                      "Train loss: {:6f}".format(loss),
                      "Train acc: {:.6f}".format(acc))
            # Compute validation loss at every 25 iterations
            if (iteration%25 == 0):
                # Initiate for validation set
                val_state = sess.run(cell.zero_state(batch_size, tf.float32))
                val_acc_ = []
                val_loss_ = []
                for x_v, y_v in get_batches(X_vld, y_vld, batch_size):
                    # Feed
                    feed = {inputs_ : x_v, labels_ : y_v, keep_prob_ : 1.0, initial_state : val_state}
                    # Loss
                    loss_v, state_v, acc_v = sess.run([cost, final_state, accuracy], feed_dict = feed)
                    val_acc_.append(acc_v)
                    val_loss_.append(loss_v)
                # Print info
                print("Epoch: {}/{}".format(e, epochs),
                      "Iteration: {:d}".format(iteration),
                      "Validation loss: {:6f}".format(np.mean(val_loss_)),
                      "Validation acc: {:.6f}".format(np.mean(val_acc_)))
                # Store
                validation_acc.append(np.mean(val_acc_))
                validation_loss.append(np.mean(val_loss_))
            # Iterate 
            iteration += 1
    saver.save(sess,"checkpoints-crnn/har.ckpt")
