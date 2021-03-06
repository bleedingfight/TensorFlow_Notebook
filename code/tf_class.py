#tensorflow 1.2.1
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
tf.set_random_seed(0)
np.random.seed(0)
n_data = np.ones((100,2))
x0 = np.random.normal(2*n_data,1)
y0 = np.zeros(100)
x1 = np.random.normal(-2*n_data,1)
y1 = np.ones(100)
x = np.vstack((x0,x1))
y = np.hstack((y0,y1))
plt.scatter(x[:,0],x[:,1],c=y,s=100,lw=0,cmap = 'RdYlGn')
plt.show()

tf_x = tf.placeholder(tf.float32,x.shape)
tf_y = tf.placeholder(tf.float32,x.shape)

l1 = tf.layers.dense(tf_x,10,tf.nn.relu)
output = tf.layers.dense(l1,2)

loss = tf.losses.sparse_softmax_cross_entropy(labels=tf_y,logits=output)
accuracy = tf.metrics.accuracy(          # return (acc, update_op), and create 2 local variables
            labels=tf.squeeze(tf_y), predictions=tf.argmax(output, axis=1),)[1]
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.05)
train_op = optimizer.minimize(loss)
sess = tf.Session()
init_op = tf.group(tf.global_variable_initilizer(),tf.local_variable_initilizer())
plt.ion()
for step in range(100):
    _,acc,pred = sess.run([train_op,accuracy,output],{tf_x:x,tf_y:y})
    if step%2 == 0:
        plt.cla()
        plt.scatter(x[:,0],x[:,1],c=pred.argmax(1),s=100,lw=0,cmap='RdYlGn')
        plt.text(1.5,-4,'Accuracy=.2f'%acc,fontdict={'size':20,'color':'red'})
        plt.pause(0.1)
plt.ioff()
plt.show()
