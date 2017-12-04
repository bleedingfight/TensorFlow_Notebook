import tensorflow as tf
k = tf.placeholder(tf.float32)
mean_moving_normal = tf.random_normal(shape = [1000],mean=(5*k),stddev = 1)
tf.summary.histogram('normal/moving_mean',mean_moving_normal)
variance_shrinking_normal = tf.random_normal(shape=[1000],mean=0,stddev = 1-(k))
tf.summary.histogram('normal/shrinking_variance',variance_shrinking_normal)
normal_combined = tf.concat([mean_moving_normal,variance_shrinking_normal],0)
tf.summary.histogram('normal/bimodal',normal_combined)
summaries = tf.summary.merge_all()
sess = tf.Session()
writer = tf.summary.FileWriter('tf_mul_dis/')
N = 400
for step in range(N):
	k_val = step/(float(N))
	summ = sess.run(summaries,feed_dict={k:k_val})
	writer.add_summary(summ,global_step = step)
