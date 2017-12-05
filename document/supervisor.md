glabel# Supervisor：Training Helper for Day-Long Trainings
为了训练一个TensorFlow操作，你可以多次简单的运行训练操作保存训练参数的checkpoint.
对于可能训练几个小时的模型效果特别好。
要求成天训练的大模型，可能通过多个复制，需要一个更加健壮的处理：
- 利落的处理关闭和崩溃。
- 在关闭或者崩溃后能继续开始。
- 可以通过TensorBoard监控。
为了能在关闭或者崩溃后继续训练，必须时常保存checkpoint，当重启的时候，必须寻找到最近的checkpoint
再次运行前载入它。
不被TensorBoard监控，必须，训练过程必须运行时常总结操作添加返回值到时间文件给TensorBoard可视化，
tensorBoard监控时间文件和显示图report随着时间训练的过程。
`tf.train.Supervisor`提供一系列服务帮助实现健壮训练处理。
如请考虑用一些提供一个丰富的训练循环和自定义选项的supersive顶层框架：`tf.learn`是一个好的选择
注意supersive用在一些大的模型上非常有用，同时可悲用在一些唯有惩罚因子的小模型上。
## Very Simple Scenario
用supervisor的最简单的场景
- 创建一个`Supervisor`对象，传递一个保存checkpoint和summaries的路径。
- 请求一个·supervisor·会话·tf.train.Supervisor.managed_session·
- 用这个会话执行训练操作，如果`supersivor`请求停止检查每一步,
```
...create graph...
my_train_op = ...

sv = tf.Supervisor(logdir="/my/training/directory")
with sv.managed_session() as sess:
  for step in range(100000):
    if sv.should_stop():
      break
    sess.run(my_train_op)
```
## 启动Service
在一个非常简单的场景中`managed_session()`启动一些运行在他们自己线程上的服务用管理session运行你的图上操作。
如果你的图上包含名字为`global_step`的整数变量，service用他的值测量训练执行的步数。
- Checkpointing Service：保存一个图上变量的副本到logdir，如果一个变量添加到你的图上，检查点文件用`global_step`
变量值，默认运行十分钟。
- Summary service：运行所有的总结操作添加他们的输出到logdir下的时间文件，默认没两分钟运行一次。
Step count:通过查看`globale_step`变量的改变计数执行了多少步，添加总结到时间文件，报道每秒的全局步数。总结tag
是`global_step/sec`,两分钟运行一次。
- Queue Runner:如果任何`tf.train.QueueRunner`增加到图上，supervisor在自己的线程登录他们。
## checking for stop
在主训练循环中检查stop是重要且必须的。
当设置`should_stop()`条件为真在Service线程中异常被reported给supervisor，其它的线程注意到这个条件适当
的终止，主训练循环，在`managed_session()`块中，不许被检查停止或者终止条件。
注意`managed_session()`考虑到从训练循环中捕获异常给supervisor。主循环不需要对异常做任何操作仅仅需要检查
停止条件。
## Recovery
如果训练程序关闭或者崩溃，它最近的checkpoint和事件文件被丢在logdir。当你重启程序，`manager_session()`
从最近的检查点恢复图，从新在停止的地方训练。
一个时间文件被创建，如果你启动一个TensorBoard指定采集目录，它将融合两个事件的内容从checkpoint最新的全局步骤训练
## larger Model Scenario
对大多数中小规模的模型这简单的场景是满足的，大模型也许当总结服务运行时内存不足：总结操作并行的运算在训练操作中
，这可能引起内存用量达到峰值达到正常用量的两倍。
对于大模型，你可以告诉supersiveor，而不是在主训练循环中运行：构造supervi时传递`summary_op=None`
例如下面代码在训练循环中运行总结每100步
```Python
...create graph...
my_train_op = ...
my_summary_op = tf.summary.merge_all()

sv = tf.Supervisor(logdir="/my/training/directory",
                   summary_op=None) # Do not run the summary service
with sv.managed_session() as sess:
  for step in range(100000):
    if sv.should_stop():
      break
    if step % 100 == 0:
      _, summ = session.run([my_train_op, my_summary_op])
      sv.summary_computed(sess, summ)
    else:
      session.run(my_train_op)

```

## Pre-trained Model Scenario
调用`managed_session()`初始化模型中的会话，如果checkpoin是可用的模型从检查点恢复，
或者从异常出初始化。  
一个常用的模型是当用一个不同的数据集训练轻微不同的模型时载入`Pre-trained`checkpoint，
我们可以通过传递一个`init function`给supersivor载入一个预先训练的checkpoint，如果模型需要
从scratch初始化这个函数被调用，当模型可能被logdir文件里面的checkpoint覆盖的时候不传递。
载入预先定义好的模型，初始化函数需要`tf.train.Saver`对象，因此你需要创建一个`saver`,这是
一个好的想法，因为新的模型需要包含先前训练的checkpoint没有呈现的变量，这个saver必须恢复之前训练的变量，如果你用
默认的saver，你从先前的模型恢复出所有的新模型的变量是可能得到一个错误的尝试。
```
...create graph...
 # Create a saver that restores only the pre-trained variables.
 pre_train_saver = tf.Saver([pre_train_var1, pre_train_var2])

 # Define an init function that loads the pretrained checkpoint.
 def load_pretrain(sess):
   pre_train_saver.restore(sess, "<path to pre-trained-checkpoint>")

 # Pass the init function to the supervisor.
 #
 # The init function is called _after_ the variables have been initialized
 # by running the init_op.
 sv = tf.Supervisor(logdir="/my/training/directory",
                    init_fn=load_pretrain)
 with sv.managed_session() as sess:
   # Here sess was either initialized from the pre-trained-checkpoint or
   # recovered from a checkpoint saved in a previous run of this code.
   ...
```
## 运行你记得Sevices
Supervisor services，像checkpoint service运行在主训练循环线程中。你可能在不同时间
点上需要添加不同的总结数据增加到你的service，而不是通常的总结服务。  
用supervisor中的`tf.train.Supervisor.loop`方法达到这个目的，在timer上重复调用你选择的函数
知道Supervisor停止条件为真，因此它在其他service上表现的不错。
例子：每20mn调用`my_addition_summaries()`  
```
def my_additional_sumaries(sv, sess):
 ...fetch and write summaries, see below...

...
  sv = tf.Supervisor(logdir="/my/training/directory")
  with sv.managed_session() as sess:
    # Call my_additional_sumaries() every 1200s, or 20mn,
    # passing (sv, sess) as arguments.
    sv.loop(1200, my_additional_sumaries, args=(sv, sess))
    ...main training loop...
```
## Writing summaries
正如`tf.summary.FileWriter`添加时间和中介到文件中一样，supervisor总是在logdir目录
创建一个事件文件。如果你想写你自己的总结正价他们到同一个时间文件中这是一个好方法，
TensorBoard更喜欢目录中一个文件被添加到相同时间文件。  
supervisor提供一个帮助函数添加总结：`tf.train.Supervisor.summary_computed`函数。
传递一个总结操作的输出结果到这个函数，下面是一个从之前的例子函数实现`my_addition_summaries()`
```
def my_additional_sumaries(sv, sess):
  summaries = sess.run(my_additional_summary_op)
  sv.summary_computed(sess, summaries)
```
## Supervisor Reference
一个Supervisor永健简单的场景或者大模型，更多高级场景下可以通过用supervisor提供的一些选项构建。
## Checkpointing:where and when
`managed_session()`可以登录checkpointing通过下面的关键字配置`Supervisor()`结构体构造的service
- logdir：Checkpointing service创建的checkpoint的路径。如果需要这个路径被创建。传递`None`
禁用checkpointing 和总结service。
- `checkpoint_basename`:创建checkpoint文件的名字，默认是`model.ckpt`
如果模型包含一个名字为`globale_step`的整数，变量的值被添加到checkpoint文件名中。  
例如：一个global step为1234，checkpoint文件是`model.ckpt-1234`
- save_model_secs:两个checkpoin之间相隔的秒数，默认是600s或者10 min。
考虑到当你的模型崩溃的时候多少数据会丢失你可以设置`save_model_secs`值，设置0禁用checkpointing service。
如果你不传递一个值supersiveor通过调用`tf.Saver()`自动创建，自动保存和恢复你模型中的所有变量
例子：每30s用一个`Saver`和`checkpoint`  
```
...create graph...
 my_saver = tf.Saver(<only some variables>)
 sv = tf.Supervisor(logdir="/my/training/directory",
                    saver=my_saver,
                    save_model_secs=30)
 with sv.managed_session() as sess:
   ...training loop...
```
## summaries:where and when
`managed_session()`调用summary服务，每秒获取总结和执行步数。，可以痛下下面关键字配置
`Supervisor()`结构体。
- logdir:总结service创建的事件文件，如果需要目录被创建。传递`None`禁用总结services
和checkpointing Sevices
- save_summaries_secs:每个总结service运行的秒数，默认为120s(2min)
传递一个0禁用summary service。
- summary_op:获取总结的操作。
如果不指定supervier用`tf.GraphKey.SUMMARY_OP`下的第一个操作`graph collection`，
如果collection是空的，supervisor通过`tf.summary.maerge_all()`创建一个操作聚集所有的总结。
传递`None`禁用summary service.  
- global_step:用于计数golbal 步数的tensor。
如归哦不指定，supervisor用`tf.GraphKey.GLOBAL_STEP`的diyige tensorBoard·graph collection·
,
如果collection是空的，supervisor在途中寻找名字为`global_step`的整数变量。  
如果找到了。全局step tensor被用于测量训练执行的步数，注意你的讯训练操作应该为增加的global step value负责。
## 模型初始化和恢复
`managed_session()`可以管理初始化或者恢复Session，准备好运行操作的时候他可以用一个完整的初始化模型返回一个session，
当`managed_session()`被调用的时候，logdir存在一个checkpoint,模型通过载入checkpoint
被初始化否则通过调用初始化槽嘴鸥或者初始化函数选项初始化。
当没有checkpoint文件的时候模型初始化通过`Supervisor()`结构体的下面关键字完成操作
- init_op:运行初始化模型操作。
如果没有指定，supersiveor用`tf.GraphKey.INIT_OP`集合中的第一个操作。如果集合为空，
supersiveor通过调用`tf.global_global_initializer()`初始化操作。
传递None禁用初始化操作  
如果指定，调用init_op(sess),这里sess是一个管理的session如果初始化操作被用，初始化函数之后调用初始化操作。  
- local_init_op:初始化没有被保存在checkpoints的像表和本地变量的图的一部分，本地初始化操作在初始化操作和初始化
函数前被运行。
如果没有指定，supersiveor用`tf.GraphKey.INIT_OP`集合中的第一个操作。如果集合为空，
supervisor将通过调用`tf.initialize_all_tabels()`和`tf.initialize_all_local_variables()`
传递None不用本地初始化操作  
- ready_op:检查模型初始化操作。

在运行了本初始化操作，初始化操作和初始化函数后，supervisor通过运行read op验证模型是否完全被初始化
如果模型被初始化这个操作返回一个空字符串，否则模型的那个部分没有被初始化。  
如果没有指定，supersiveor用`tf.GraphKey.READ_OP`集合中的第一个操作。如果集合为空，supervisor
创建一个一个ready op验证所有的变量被`tf.report_uninitialized_variable()`初始化.
传递`None`禁用ready op,这种情况下模型初始化后不被检查。
checkpoints恢复被`Supervisor()`下的下面关键字控制。
- logdir:寻找ckeckpoint路径，checkpoint service保存原始的文件，名字为`checkpoint`,在checkpoint目录下
预示这个路径是醉经的checkpoint路径。
文件是文本格式的，当文本小的时候你可以手动编辑从不同的checkpoint文件恢复而不是最近的文件。
- read_op:read op运行在在前然后载入checkpoint,第一次运行检查模型是否被初始化第二次运行验证模型是否被完全初始化。
- local_init_op:本地初始化操作，运行在read op第一次运行前，初始化本地变量和表。
- saver：保存用于载入checkpoint的Saver对象。
