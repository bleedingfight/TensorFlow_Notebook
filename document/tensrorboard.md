# TensorBoard使用
参数：[-h] [--logdir LOGDIR] [--debug [DEBUG]] [--nodebug]
                   [--host HOST] [--inspect [INSPECT]] [--noinspect]
                   [--tag TAG] [--event_file EVENT_FILE] [--port PORT]
                   [--purge_orphaned_data [PURGE_ORPHANED_DATA]]
                   [--nopurge_orphaned_data]
                   [--reload_interval RELOAD_INTERVAL]

选项值：

- h,--help:显示帮助信息和退出。  
- --logdir LOGDIR:指定数据采集到哪儿。
- TensorBoard will:找到能被显示的TensorBoard event 文件，TensorBoard将递归的遍历logdir所在的目录结构，寻找\*tfevents.\*文件,你也可以通过传递一个都好分割采集目录的列表，TensorBoard将查看每个目录。你也可以通过放一个冒号在名字和路径之间指定单个采集目录的名字，正如在tensorboard中的--logdir=name1:/path/to/logs/1,name2:/path/to/logs/2
- --debug [DEBUG]:是否在调试状态下运行程序。这个参数增加了采集详情到DEBUG
- --nodebug:监听host，默认服务在0.0.0.0,设置到127.0.0.1（localhost）禁用远程访问(也禁用安全警告)
- --inspect [INSPECT] 用这个标志打印除你的event文件的概要到command，当TensorBoard没有数据显示或者显示数据很奇怪的时候。用法:  
```
usages:
 tensorboard --inspect --event_file=myevents.out
 tensorboard --inspect --event_file=myevents.out --tag=loss tensorboard --inspect --logdir=mylogdir
 tensorboard --inspect --logdir=mylogdir --tag=loss See
tensorflow/python/summary/event_file_inspector.py for
more info and detailed usage.
```
-  --noinspect
-  --tag TAG            一个类似的查询标签，仅被用在 --inspect参数一起使用时。
- --event_file EVENT_FILE：一个类似的用于查询的事件文件，仅当--inspect被present和--logdir没有指定时候使用。
- --port PORT           Tensorboard服务器工作的端口。
- --purge_orphaned_data [PURGE_ORPHANED_DATA]：是否清除可能会导致TensorBoard重启的数据，禁用purge_orphaned_data可能调制数据消失。
- --nopurge_orphaned_data
- --reload_interval RELOAD_INTERVAL：后台载入数据的频率。
