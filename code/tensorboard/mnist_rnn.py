import tensorflow as tf
from tensorflow.contrib import rnn
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/",one_hot=True)
time_steps = 28
num_units = 128
n_input = 28
learning_rate = 0.001
n_classes = 10
batch_size = 128
out_weights = tf.Variable(tf.random_normal([num_units,n_classes]))
out_bias = tf.Variable(tf.random_normal([n_classes]))

x = tf.placeholder("float",[None,time_steps,n_input])
y = tf.placeholder("float",[None,n_classes])
input= tf.unstack(x ,time_steps,1)
lstm_layer = rnn.BasicLSTMCell(num_units,forget_bias = 1)
outputs,_ = rnn.static_rnn(lstm_layer,input,dtype="float32")
prediction = tf.matmul(outputs[-1],out_weights)+out_bias
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = prediction,labels = y))
opt = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(loss)
correct_prediction = tf.equal(tf.argmax(prediction,1),tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    iter = 1
    while iter<800:
        batch_x,batch_y = mnist.train.next_batch(batch_size = batch_size)
        batch_x = batch_x.reshape((batch_size,time_steps,n_input))
        sess.run(opt,feed_dict = {x:batch_x,y: batch_y})
        if iter %10 == 0:
            acc = sess.run(accuracy,feed_dict = {x:batch_x,y:batch_y})
            los = sess.run(loss,feed_dict = {x:batch_x,y:batch_y})
            print("For iter ",iter)
            print("Accuracy ",acc)
            print("Loss ",los)
        iter = iter+1
    test_data = mnist.test.images[:128].reshape((-1,time_steps,n_input))
    test_label = mnist.test.labels[:128]
    print("Test accuracy:",sess.run(accuracy,feed_dict={x:test_data,y:test_label}))

