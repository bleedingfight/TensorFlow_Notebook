# TensorFLow中的队列
![](http://img0.ph.126.net/0Z9_-AiPBjpxBQ4dmOTC4A==/6631842921399089997.gif)
```Python
import tensorflow as tf
q = tf.FIFOQueue(3,'float')     #1
init = q.enqueue_many([0.0,0.0,0.0]) #2
x = q.dequeue()                 #3
y = x+1                         #4
q_inc = q.enqueue([y])          #5
q_inc.run()                     #6
q_inc.run()                     #7
q_inc.run()                     #8

```
代码中的Enqueue, EnqueueMany, and Dequeue表示的是特殊的节点，他们获取的是指针而不是真正的值。  
这段代码，我们先建立一个存放三个数据的先进先出的队列（1），队列中放的数据是浮点数，然后通过队列的初始化方法，将队列里面的数据赋值为0（2）,然后取出队列里面的元素（3），将队列里面的元素加1，后复制给变量y（4），然后将y加入队列（5），然后启动图操作，每次启动图都会运行图，最终队列里面的元素将全部变为1.
整体执行分为两步，第一步是构建图，第二步是操作。  
TensorFlow中的Session对象是多线程的，多线程可以很容易的实现在同一个session中并行运算操作。但是真正实现却不是如此简单。所有的线程必须同时停止，异常必须能被捕获和report，队列必须在stoping的shibuichi被正确的关闭。

TensorFlow提供两个类，tf.train.Coordinator和tf.train.QueueRunner,这两个类被一起设计，Coordinator类帮助多线程同时停止，等待他们结束。report异常给程序，QueueRunner被用在统一队列来同时创建一些线程到队列tensors中  
## Coordinator
- tf.train.Coordinator.should_stop: returns True if the threads should stop.
- tf.train.Coordinator.request_stop: requests that threads should stop.
- tf.train.Coordinator.join: waits until the specified threads have stopped.
```Python
# Thread body: loop until the coordinator indicates a stop was requested.
# If some condition becomes true, ask the coordinator to stop.
def MyLoop(coord):
  while not coord.should_stop():
    ...do something...
    if ...some condition...:
      coord.request_stop()

# Main thread: create a coordinator.
coord = tf.train.Coordinator()

# Create 10 threads that run 'MyLoop()'
threads = [threading.Thread(target=MyLoop, args=(coord,)) for i in xrange(10)]

# Start the threads and wait for all of them to stop.
for t in threads:
  t.start()
coord.join(threads)
```
## QueueRunner

QueueRunner创建一些此线程，重复执行入队操作。这些线程可被用一个协调者来一起stop。另外，一个queue runner 运行一个closer线程，如果协调者得到异常将自动关闭队列。
实现上面的描述：
首先用TensorFlow queue建立一个图(例如tf.RandomShuffleQueue),增加操作处理这个例子，入队。增加训练操作开始出队。
```Python
example = ...ops to create one example...
# Create a queue, and an op that enqueues examples one at a time in the queue.
queue = tf.RandomShuffleQueue(...)
enqueue_op = queue.enqueue(example)
# Create a training graph that starts by dequeuing a batch of examples.
inputs = queue.dequeue_many(batch_size)
train_op = ...use 'inputs' to build the training part of the graph...
```
在Python训练程序中，创建一个Runner将运行几个线程处理examples入队，创建一个Coordinator，要求queue runner随着协调者开始它的线程，用协调者写一个训练循环
```Python
# Create a queue runner that will run 4 threads in parallel to enqueue
# examples.
qr = tf.train.QueueRunner(queue, [enqueue_op] * 4)

# Launch the graph.
sess = tf.Session()
# Create a coordinator, launch the queue runner threads.
coord = tf.train.Coordinator()
enqueue_threads = qr.create_threads(sess, coord=coord, start=True)
# Run the training loop, controlling termination with the coordinator.
for step in xrange(1000000):
    if coord.should_stop():
        break
    sess.run(train_op)
# When done, ask the threads to stop.
coord.request_stop()
# And wait for them to actually do it.
coord.join(enqueue_threads)

```
## 处理异常
被queue runner线程启动线程不仅仅运行一些入队操作，他们捕获，处理队列生成的异常，包括report队列被关闭的tf.errors.OutOfRangeError异常
一个用协调者捕获或者report异常的训练程序应该跟下面类似。
```Python
try:
    for step in xrange(1000000):
        if coord.should_stop():
            break
        sess.run(train_op)
except Exception, e:
    # Report exceptions to the coordinator.
    coord.request_stop(e)
finally:
    # Terminate as usual. It is safe to call `coord.request_stop()` twice.
    coord.request_stop()
    coord.join(threads)
```
