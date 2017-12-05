# TensorFlow Version Semantics
## 语法2.0
TensorFlow语法2.0对于它的公共API，每个释放的版本都有`MAJOR.MINOR.PATCH`
- MAJOR:向后不兼容的改变，代码和数据将不需要在新的版本下工作，然而在一些情况下存在TensorFlow数据（图，checkpoint，和其它的Protobufs）
也许被移植到新版本上，下面由关于数据兼容性的详细介绍。
- MINOR:向后兼容特性，加速。等等，代码和数据工作在之前的版本仅仅依靠公共的API将继续工作，不需要改变，更多详细的不能在公共API工作的看下面。
- PATCH向后兼容bug修复。

## 覆盖了什么
仅仅公共API，包括`tf.contrib`,这包括所有额公共函数和类（他们的名字不是以——开头）
在tensorflow模型和它的子模型中。注意`examples/`和`tools/`目录下的代码通过tensorflow的Python模型不能到达，这样通过兼容性保证不能覆盖。
一个符号通过`tensorflow`Python模型或者它的子模型可用，但是没有被记录，它就不被考虑作为公共API的一部分。
- C API
- peotocal buffer file:attr_value, config, event, graph, op_def, reader_base, summary, tensor, tensor_shape, and types.
## 什么没有被覆盖
有一些API函数是很有用的，被标记为实验性的，可能在后续不兼容情况改善。包括：
- 实验API，在Python中`tf.contrb`模型和它的子模型和一些函数在C的API或者在protocol buffers是明确的标注为实验性的。
- 其他语言：Tensorflow API在其他语言上除了Python和C还有：
- C++
- Java
- Go
- 详细的组合操作：一些Python中的公共函数扩展为图中的一些主要操作，这些详细的将称为图的一部分以`GraphDefs`保存在磁盘上,这些详细的允许改较小的gambian，类似，回归测试
测试检查图和确切的匹配。
在API中一些方法标记为`deprecated`,可能在一些小版本更新中被删除。
