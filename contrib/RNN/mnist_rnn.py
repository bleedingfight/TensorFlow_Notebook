import tensorflow as tf
from tensorflow.contrib import rnn
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/",one_hot=True)

# define constants
time_steps=28
# hidden LSTM units
num_units=128
#rows of 28 pixels
n_inputs=28
# learning rate for adam
learning_rate=0.001
# mnist is meant to be classified in 10 classes(0-9).
n_classes=10
# size of batch
batch_size=128
# weights and biases of appropriate shape to accomplish above task
with tf.name_scope("Last_layer") as scope:
    out_weights=tf.Variable(tf.random_normal([num_units,n_classes]),name='out_weight')
    out_bias=tf.Variable(tf.random_normal([n_classes]),name='out_bias')
    w_hist = tf.summary.histogram('weights',out_weights)
    b_hist = tf.summary.histogram('bias',out_bias)
    # input image placeholder
    x=tf.placeholder("float",[None,time_steps,n_inputs])
    # input label placeholder
    y=tf.placeholder("float",[None,n_classes])
# processing the input tensor from [batch_size,n_steps,n_input] to "time_steps" number of [batch_size,n_input] tensors
inputs=tf.unstack(x ,time_steps,1)
# defining the network
lstm_layer=rnn.BasicLSTMCell(num_units,forget_bias=1)
outputs,_=rnn.static_rnn(lstm_layer,inputs,dtype="float32")
    # converting last output of dimension [batch_size,num_units] to [batch_size,n_classes] by out_weight multiplication
with tf.name_scope("Evaluate_op") as scope:
    prediction=tf.matmul(outputs[-1],out_weights)+out_bias
    # loss_function
    loss=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction,labels=y))
    # optimization
    tf.summary.scalar("loss",loss)
opt=tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

# model evaluation
correct_prediction=tf.equal(tf.argmax(prediction,1),tf.argmax(y,1))
accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
# initialize variables
init=tf.global_variables_initializer()
summary = tf.summary.merge_all()
with tf.Session() as sess:
    sess.run(init)
    writer = tf.summary.FileWriter('./mnist_tb')
    writer.add_graph(sess.graph)
    iter=1
    while iter<800:
        batch_x,batch_y=mnist.train.next_batch(batch_size=batch_size)
        batch_x=batch_x.reshape((batch_size,time_steps,n_inputs))
        sess.run(opt, feed_dict={x: batch_x, y: batch_y})
        if iter %10==0:
            acc=sess.run(accuracy,feed_dict={x:batch_x,y:batch_y})
            los=sess.run(loss,feed_dict={x:batch_x,y:batch_y})
            s,_ = sess.run([summary,opt],feed_dict={x:batch_x,y:batch_y})
            writer.add_summary(s,global_step=iter)
            print("For iter:",iter,"Accuracy:",acc,"Loss",los)
        iter=iter+1
    # calculating test accuracy
    test_data = mnist.test.images[:128].reshape((-1, time_steps, n_inputs))
    test_label = mnist.test.labels[:128]
    print("Testing Accuracy:", sess.run(accuracy, feed_dict={x: test_data, y: test_label}))        
