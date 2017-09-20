#TensorFlow 1.3 GPU ubuntu14.0.4调试通过
import tensorflow as tf
filename = ['./t1.txt','./t2.txt']
filename = tf.constant(filename)
dataset = tf.contrib.data.Dataset.from_tensor_slices(filename)
dataset = dataset.flat_map(lambda filename:(
tf.contrib.data.TextLineDataset(filename).skip(1).filter(lambda line:tf.not_equal(tf.substr(line,0,1),'#'))))
iterater = dataset.make_one_shot_iterator()
next_element = iterater.get_next()
sess = tf.Session()
for i in range(10):
     print(sess.run(next_element))


