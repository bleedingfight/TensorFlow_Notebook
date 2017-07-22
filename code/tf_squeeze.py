import tensorflow as tf
a = tf.constant([1,2,1,3,1,1])
b = tf.squeeze(a)
with tf.Session() as sess:
    print((sess.run(b))
