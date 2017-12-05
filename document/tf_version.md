# TensorFlow数据版本：GraphDefs和Checkpoints
正如在`Compatiability for Graphs and CheckPoints`中描述的，TensorFlow标记
每种数据版本信息为了维保证向后兼容，这边温江提供了详细的版本机制，和如何安全的改版数据格式。
## 向后兼容和部分向前兼容
两个在TensorFlow核心的输出和输入时checkpoint（xuliehaude变量状态）
和`GraphDefs`(序列化的计算图)，任何人工版本化方法必须高率如下的要求：
- 向后兼容支持载入老版本的TensorFlow的`GraphDefs`
- 向前兼容支持更新新版本的TensorFlow的`GraphDef`
- 改进TensorFlow不兼容的,例如删除操作，增加属性和移除属性
对于`GraphDefs`，在major版本中保持向后兼容，这意味着函数可能被移除在major版本，
向前兼容被强制在Patch release（1.x.1->1.x.2,例如）
为了获得前后兼容性正如强制改变格式，xuiehuabiaoshi图和变量状态需要有元数据，这节下面详细的TensorFLow实现
和指导`GraphDef`版本。
