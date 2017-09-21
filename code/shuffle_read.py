import tensorflow as tf
filename = ['train_x.txt','train_y.txt']
filename = tf.constant(filename)
dataset = tf.contrib.data.TextLineDataset(filename)
dataset = dataset.batch(4)
dataset = dataset.shuffle(100,seed=0)
iterator = dataset.make_one_shot_iterator()
next_element = iterator.get_next()
sess = tf.Session()
data = []
try:
     while True:
          data.append(sess.run(next_element))
   except:
      pass
   n = len(data)
    print("读取数据个数",n)
   for i in data:
       print(data)

           
