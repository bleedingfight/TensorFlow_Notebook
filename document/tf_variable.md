# 变量的创建，初始化，保存，载入
当你训练一个模型，你用`variable`保持，更新参数，Variable在in-memory buffers，包含有tensor。他们必须被合理的初始化才在训练后能被保存到硬盘。你可以之后恢复值取训练和分析模型。
这篇文章用到下面两个类
- The tf.Variabe class
- The tf.train.Saver class
## 创建
当你创建一个`Variable`，你传递一个Tensor作为结构体`Variable()`的初始化值。TensorFlow提供了一系列的初始化操作。
注意所有的操作都要求你指定tensor的形状，形状将变为variable的形状，Variable通常有一个固定的形状，但是TensorFlow提供一个更高级的机制改变变量的形状。
```Python
# Create two variables.
weights = tf.Variable(tf.random_normal([784, 200], stddev=0.35),
                      name="weights")
biases = tf.Variable(tf.zeros([200]), name="biases")
```
调用tf.Variable()增加一些操作到图上：
- 一个`variable`操作保持变量的值。
- 初始化操作设置变量为初始化的值，通常就是一个tf.assign操作。
- 对于初始化值，像例子中对`biases`变量的`zeros`增加到了图上。
## Devie placement
一个变量可以在它被创建的时候依附到一个特别的device，用`with tf.device(...):`块
```Python
# Pin a variable to CPU.
with tf.device("/cpu:0"):
  v = tf.Variable(...)

# Pin a variable to GPU.
with tf.device("/gpu:0"):
  v = tf.Variable(...)

# Pin a variable to a particular parameter server task.
with tf.device("/job:ps/task:7"):
  v = tf.Variable(...)
```
操作，改变变量，像`tf.Variable.assign`和`tf.train.Optimizer`的
参数更新操作必须运行在同一作为变量的device创建这些操作时不兼容的placement将被忽略
Device placement在运行一个重复的设置时很重要，查看'tf.train.repica_device_setter'了解replicated 模型简化devices的简化配置。
## 初始化
在你的模型运行前变量必须被合理的初始化，一个简单的方法是在运行模型前增加操作运行所有变量的初始化器，你可以从checkpoit file恢复变量。
查看如下：
用`tf.global_variable_initilizer()`正佳一个操作运行变量初始化器，仅仅这个操作后你的模型在launch session才有完整的结构。
```Python
# Create two variables.
weights = tf.Variable(tf.random_normal([784, 200], stddev=0.35),
                      name="weights")
biases = tf.Variable(tf.zeros([200]), name="biases")
...
# Add an op to initialize the variables.
init_op = tf.global_variables_initializer()

# Later, when launching the model
with tf.Session() as sess:
  # Run the init operation.
  sess.run(init_op)
  ...
  # Use the model
  ...

```
## 从另一个变量初始化
你有时候需要从其他变量初始化一个变量，当你需要这种初始化时你必须小心，因为`tf.global_variables_initializer()`
同时初始化变量。
从另一个变量初始化新的变量用其他变量的`initialized_value()`属性，你可以直接初始化值作为你新编量的值
或者你可以用另一个tensor为新的变量计算一个值。
```Python
# Create a variable with a random value.
weights = tf.Variable(tf.random_normal([784, 200], stddev=0.35),
                      name="weights")
# Create another variable with the same value as 'weights'.
w2 = tf.Variable(weights.initialized_value(), name="w2")
# Create another variable with twice the value of 'weights'
w_twice = tf.Variable(weights.initialized_value() * 2.0, name="w_twice")
```
## 自定义初始化
你可以传递一个合适的变量给`tf.variable_initializer`,查看`Variables Dcumenttation`获取更多选项。
## 保存和恢复
最简单保存恢复一个模型是用`tf.train.Saver`对象，结构体增加`save`和`restore`操作
到图上，或者图上变量一个特殊的列表`saver`对象提供方法运行这些操作，指定读写checkpoint文件的路径。
## Checkpoint Files
变量被保存在二进制文件中，包含一个从变量名字到tensor的映射
你可以创建一个`Saver`对象，你产品你可以从checkpoint文件选择变量的名字，默认，它用`tf.Variable.name`属性为每个变量。
理解什么是checkpoint里面的变量是什么，你可以用`inspect_checkpoint`库和`print_tensors_in_checkpoint_file`函数

## 保存变量
用`tf.train.Saver()`创建一个`Saver`管理模型中所有的变量
```pyhton
# Create some variables.
v1 = tf.Variable(..., name="v1")
v2 = tf.Variable(..., name="v2")
...
# Add an op to initialize the variables.
init_op = tf.global_variables_initializer()

# Add ops to save and restore all the variables.
saver = tf.train.Saver()

# Later, launch the model, initialize the variables, do some work, save the
# variables to disk.
with tf.Session() as sess:
  sess.run(init_op)
  # Do some work with the model.
  ..
  # Save the variables to disk.
  save_path = saver.save(sess, "/tmp/model.ckpt")
  print("Model saved in file: %s" % save_path)
```
## 恢复变量
恢复和用于保存的`Saver`对象一样，注意当你从文件恢复变量之前不要初始化他们。
```Python
# Create some variables.
v1 = tf.Variable(..., name="v1")
v2 = tf.Variable(..., name="v2")
...
# Add ops to save and restore all the variables.
saver = tf.train.Saver()

# Later, launch the model, use the saver to restore variables from disk, and
# do some work with the model.
with tf.Session() as sess:
  # Restore variables from disk.
  saver.restore(sess, "/tmp/model.ckpt")
  print("Model restored.")
  # Do some work with the model
  ...

```
## 选择保存，恢复的变量
如果你传递一个参数给`tf.train.Saver()`,`saver`处理图中所有的变量。
每一个变量被保存在变量被创建时的名字下面。
它有时候在指定checkpoint文件变量的名字时很管用。例如，你也许训练一个模型，变量命名为`weights`，
你想恢复它到新的名字`params`下。
当保存或者恢复模型中变量的一个子集时非常有用。例如，你也许训练一个5层的神经网络，你现在练一个新的模型
6层模型，恢复之前训练的5层的参数到一个新的6层，从5层保存参数到新的模型的前5层。
你可以通过传递`tf.train.Saver()`一个Python字典很容易指定保存的名字和变量，值时变量，键时名字。
注意:
- 如果你需要保存模型变量不同的子集你可以创建一些saver对象。相同的变量可能在多个saver对象中，它的值仅仅在saver的restore()放法运行时改变
- 在开启session后如果仅仅恢复变量的一个子集，你必须为所有的变量运行一个initial操作。看`tf.variables_initializer`获取更多信息。
```Python
# Create some variables.
v1 = tf.Variable(..., name="v1")
v2 = tf.Variable(..., name="v2")
...
# Add ops to save and restore only 'v2' using the name "my_v2"
saver = tf.train.Saver({"my_v2": v2})
# Use the saver object normally after that.
...
```
