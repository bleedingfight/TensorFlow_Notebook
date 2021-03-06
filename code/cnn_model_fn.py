def cnn\_model\_fn(features,labels,mode):
    # Input Layer
    input_layer = tf.reshape(features["x"],[-2,28,28,1])
    #Convolutional layer1
    conv1 = tf.layers.conv2d(
	inputs=input_layers\,
	filters=32,
	kernel_size=[5,5],
	padding="same",
	activation=tf.nn.relu)
    pool1 = tf.layers.max_pool2d(inputs=conv1,pool_se=[2,2],strides=2)
    conv2 = tf.layers.conv2d(
	inputs=pool,
	filters=64,
	kernel_size=[5,5],
	padding='same',
	activation=tf.nn.relu)
    pool2 = tf.layers.max_pooling2d(inputs=conv2,pool_size=[2,2],strides=2)
    dense = tf.layers.dense(inputs=pool2_flat,units=1024,activaton=tf.nn.relu)
    dropout = tf.layers.dense(inputs=dropout,units=10)
    predictions = {'classes':tf.argmax(input=logits,axis=1),
		'probabilities':tf.nn.softmax(logits,name='softmax_tensor')
		}
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode,predictions=predictions)
    one_hot_labels = tf.one_hot(indices=tf.case(labels,tf.int32),depth=10)
    loss = tf.losses.softmax_cross_entropy(
	onehot_labels=onehot_labels,logits=logits)
    if mode = tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
        train_op = optimizer.minimize(loss=loss,global_step=tf.train.get_global_step())
    return tf.estimator.EstimatorSpec(mode=mode,loss=loss,train_op=train_op)
    eval_metric_op = {
	'accuracy':tf.metrics.accuracy(labels=labels,prediction=predictions['classes'])}
    return tf.estimator.EstimatorSpec(
	mode=mode,loss=loss,eval_metric_ops=eval_metric_ops)
