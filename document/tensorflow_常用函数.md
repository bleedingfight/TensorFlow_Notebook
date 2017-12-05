# 函数方法(表一)
|tf下的方法||tf下的方法||tf下的方法||
|:---:|:----:|:---:|:---:|
|abs||batch_space||convert_to_tensor_or_indexed_slices||
|accumulate_n||batch_to_space_nd||convert_to_tensor_or_sparse_tensor||
|acos||betainc||cos||
|add||bfloat16||count_nonzero||
|add_check_numerics_ops||bitcast||count_up_to||
|add_n||bool||create_partitioned_variables||
|add_to_collection||boolean_mask||cross||
|AggregationMethod||broadcast_dynamic_shape||cumprod||
|all_variables||case||cumsum||
|app||cast||decode_base64||
|argmax||ceil||decode_csv||
|arg_min||check_numbers||decode_json_example||
|argmax||cholesky||decode_raw||
|argmin||cholesky_solve||delete_session_tensor||
|as_dtype||clip_by_average_norm||depth_to_space||
|as_string||clip_by_global_norm||dequantize||
|asin||clip_by_global_norm||deserialize_many_sparse||
|Assert||clip_by_value||device||
|assert_equal||compat||DeviceSpec||
|assert_greater||COMILER_VERSION||diag||
|assert_greater_equal||complex||diag_part||
|assert_integer||complex128||digamma||
|assert_less_equal||complex64||div||
|assert_negative||concat||divide||
|assert_non_negative||cond||double||
|assert_non_positive||ConditionalAccumulator||DType||
|assert_proper_iterable||ConditionAccumulatorBase||dynamic_partition||
|assert_rank||ConfigProto||synamic_stich||
|assert_rank_at_least||confusion_matrix||edit_distance||
|assert_type||conj||einsum||
|assert_variables_initialized||constant||encode_base64||
|assign||constant_initializer||equal||
|assign_add||container||erf||
|assign_sub||contrib||erfc||
|atan||control_dependencies||errors||
|AttrValue||convert_to_tensor||Event||
# 函数方法(表二)
|tf下的方法||tf下的方法||tf下的方法||
|:---:|:----:|:---:|:---:|
|exp||get_variable||int32||
|expand_dims||get_variable||ini64||
|expml||get_variable_scope||int8||
|extract_image_patches||gfile||InteractiveSession||
|eye||GIT_VERSION||invert_permutation||
|fake_quant_with_min_max_args||global_norm||is_finite||
|fake_quant_with_min_max_args_gradient||global_variables||is_inf||
|fake_quant_with_min_max_vars||gloabal_variable_initializer||is_nan||
|fake_quant_with_min_max_vars_gradient||GPUOption||is_non_decreasing||
|fake_quant_with_min_max_vars_per_channel||gradients||is_numeric_tensor||
|fake_quant_with_min_max_vars_per_channel_gradient||Graph||is_strictly_increasing||
|fft||GRAPH_DEP_VERSION||is_variable_initialized||
|fft2d||GRAPH_DEP_VERSION_MIN_CONSUMER||layers||
|fft3d||GRAPH_DEP_VERSION_MIN_PRODUCE||lbeta||
|FIFOQueue||graph_util||less||
|fill||GraphDef||less_equal||
|fixed_size_partitioner||GraphKeys||lgamma||
|FixedLenFeature||GraphOptions||lin_space||
|FixedLengthRecordReader||greater||load_file_system_library||
|FixedLenSequenceFeature||greater_equal||load_op_library||
|flags||group||local_variables||
|float16||half||local_variables_initializer||
|float16||hessions||log||
|float64||histogram_fixed_width||loglp||
|floor||HistogramProto||logging||
|floor_div||identity||logical_and||
|floordiv||IdentityReader||logical_not||
|floormod||ifft||logical_or||
|foldl||ifft2d||logical_xor||
|foldr||ifft3d||LogMessage||
|gather||igamma||losses||
|gather_nd||igammac||make_template||
|get_collection||imag||map_fn||
|get_collection_ref||image||matching_files||
|get_default_graph||import_graph_def||matmul||
|get_default_session||IndexedSlices||matrix_hand_part||
|get_local_variable||initialize_all_tabels||matrix_determinant||
|get_seed||initialize_all_variable||matrix_diag||
|get_session_handle||initialize_local_variable||matrix_diag_part||
|get_session_tensor||ini16||matrix_inverse||

# 表三

|tf下的方法||tf下的方法||tf下的方法||
|:---:|:----:|:---:|:---:|
|matrix_set_diag||parse_tensor||reduce_prod||segment_mean|
|matrix_solve||placeholder||reduce_sum||segment_prod|
|matrix_solve_ls||placeholder_with_default||register_tensor_conversion_function||segment_sum|
|matrix_transpose||polygamma||Print||RegisterGradient|
|matrix_triangular_solve||pow||PriorityQueue||report_uninitialized_varables|
|maxmin||py_func||required_space_to_batch_paddings||selt_adjoint_eig|
|meshgrid||python_io||reset_default_graph||self_adjoit_eigvals|
|metrics||pywrap_tensorflow||reshape||sequence_mask|
|min_max_variable_partitioner||qint16||resource||serialize_many_sparse|
|minmax||qint32||resource_loader||serialize_sparse|
|mod||qint8||resverse||Session|
|model_variable||qr||reverse_sequence||Sessionlog|
|moving_average_variables||quantize_v2||reverse_v2||set_random_seed|
|multinomial||quantized_concat||rint||setdiff1d|
|multiply||QUANTIZED_DIVPES||round||sets|
|name_scope||QueueBase||rsqrt||shape|
|nameAttrList||quint16||RunMetadata||shape_n|
|negative||quint8||RunOptions||sigmoid|
|newaxis||random_crop||staurate_cast||sign|
|nn||random_normal||saved_model||sin|
|no_op||random_normal_initializer||scale_mul||size|
|no_regularizer||random_shffleQueue|scatter_add|||slice|
|tf.NodeDef||range||scatter_div||space_to_batch|
|NoGradient||rank||scatter_mul||space_to_batch_nd|
|norm||read_file||scatter_nd||space_to_depth|
|not_equal||ReaderBase||scatter_nd_add||sparse_add|
|NotDifferentiable||real||scatter_nd_sub||sparse_concat|
|one_hot||realdiv||scatter_nd_update||spare_fill_enpty_rows|
|ones||reciprocal||scatter_sub||sparse_mask|
|ones_initializer||reduce_all||scatter_update||sparse_matmul|
|pad||reduce_any||sdca||sparse_maximum|
|PaddingFIFOQueue||reduce_join||segment_max||sparse_merge|
|parallel_stack||reduce_logsumexp||segment_mean||sparse_minimum|
|parse_example||reduce_max||segment_min||sparse_placeholder|
|parse_single_example||reduce_mean||segment_prod||sparse_reduce_sum|
|parse_sigle_sequence_example||reduce_min||segment_sum||sparse_reduce_sum_sparse|

# 表三
|tf下的方法||tf下的方法||tf下的方法||
|:---:|:----:|:---:|:---:|
|sparse_recorder||transpose||||
|sparse_reset_shape||truediv||||
|sparse_reshape||truncated_normal||||
|sparse_retain||truncated_normal_initilizer||||
|sparse_segment_mean||truncatediv||||
|sparse_segment_sqrt_n||truncatemod||||
|sparse_segment_sum||tuple||||
|sparse_softmax||uint16||||
|sparse_split||uint8||||
|sparse_tensor_dense_matmul||uniform_unit_scaling_initializer||||
|sparse_tensor_to_dense||unique||||
|sparse_to_dense||unique_with_counts||||
|sparse_to_indicator||unsorted_segment_sum||||
|sparse_transpose||unstack||||
|SparseConditionalAccumulator||user_ops||||
|SparseFeature||Variable||||
|SparseTensor||variable_axis_size_partitioner||||
|SparseTensorValue||variable_op_scope||||
|split||variable_scope||||
|sqrt||variable_initializer||||
|square||VarLenFeature||||
|squared_difference||verify_tensor_all_finite||||
|squeeze||VERSION||||
|stack||where||||
|stop_gradient||while_loop||||
|strided_slice||WholeFileReader||||
|string||write_file||||
|string_join||zeros||||
|string_tp_hash_bucket||zeros_initializer||||
|string_to_hash_bucket_fast||zeros_like||||
|string_to_number||zeta||||
|substr||||||
|subtract||||||
|summary||||||
|Summary||||||
|svd||||||
|sysconfig||||||
|tabels_initilizer||||||
|tan||||||
|tanh||||||
|Tensor||||||
|TensorArray||||||
|tensordot||||||
|TensorInfo||||||
|TensorShape||||||
|TextLineReader||||||
|TFRecordReader||||||
|title||||||
|to_bfloat16||||||
|to_double||||||
|to_float||||||
|to_int32||||||
|to_int64||||||
|trace||||||
|train||||||
|trainable_variables||||||

## softmax_cross_entropy_with_logits
softmax_cross_entropy_with_logits(_sentinel=None, labels=None, logits=None, dim=-1, name=None)
计算Logits和Labels的softmax交叉熵，衡量在离散分类任务相互之间的错误概率（每一个熵是一类），例如CIFAR-10图片仅仅有一个标签，一个图片可能是一只狗也可能是一个拖拉机，但是不是同时是狗或者拖拉机。
注意，每个类是互斥的，他们的概率不需要是。所有的这些要求每一行Labels时一个可用的概率分布，如果他们不是，计算的梯度将不正确。
如果用互斥的labels(同一时间只有一个类时正确的)。  
**警告**:操作期望不能比例的Logits，因为它内部在Logits上高效的执行softmax，不要用这个操作softmax的输出，这样产生错误的结果。
Logits和‘Label’不许由相同的形状，[batch_size,num_classes]和相同的数据类型(float16,float32或者float64)。  
**注意避免冲突，它要求传递参数的名字给函数**
### Args:
- _sentinel:阻止位置参数，不常用。
- Labels:labels[i]的每一个参数必须是一个可用的概率分布。
- dim:类别的维度。，默认-1时last维。
- name:操作的名字。  
 Returns
返回一个batch_size长度的logits和softmax交叉熵的损失的1维的Tensor。
      A 1-D `Tensor` of length `batch_size` of the same type as `logits` with the
      softmax cross entropy loss.
### argmax
属于tensorflow.python.ops.math_ops:
argmax(input, axis=None, name=None, dimension=None)
返回tensor一个轴上的最大值的索引。    
 Args:
- input，A（Tensor）必须时如下数据类型中的一种， `float32`, `float64`, `int64`, `int32`, `uint8`, `uint16`, `int16`, `int8`, `complex64`, `complex128`, `qint8`, `quint8`, `qint32`, `half`.
- axis: A `Tensor`. 必须时下面数据类型中的一种 `int32`, `int64`.
        int32, 0 <= axis < rank(input). 描述输入Tensor到reduce across的轴，对于向量，用 axis = 0 Describes which axis
        of the input Tensor to reduce across. For vectors, use axis = 0.
- name: 操作的一个名字。

- Returns:返回一个int64的Tensor
### cast
 tensorflow.python.ops.math_ops:

cast(x, dtype, name=None)
转换一个tensor到新的类型，操作casts    
    For example:

    ```python
    # tensor `a` is [1.8, 2.2], dtype=tf.float
    tf.cast(a, tf.int32) ==> [1, 2]  # dtype=tf.int32
    ```

- Args:
      x: A `Tensor` or `SparseTensor`.  
      dtype: The destination type.  
      name: A name for the operation (optional).  

- Returns:
      A `Tensor` or `SparseTensor` with same shape as `x`.

    Raises:
      TypeError: If `x` cannot be cast to the `dtype`.
### reduce_mean
tensorflow.python.ops.math_ops:
reduce_mean(input_tensor, axis=None, keep_dims=False, name=None, reduction_indices=None)
计算tensor a的元素均值  
沿着给定的axis的维度 reduce input_tensor
keep_dims是True，对于每一个axis中的输入，tensor的rank减小到1,如果keep_dims是True，antipode维度被保留长度1.
如果`axis`没有输入,所有的万恶度被减小，一个单个元素的tensor被返回。     
    For example:

    ```python
    # 'x' is [[1., 1.]
    #         [2., 2.]]
    tf.reduce_mean(x) ==> 1.5
    tf.reduce_mean(x, 0) ==> [1.5, 1.5]
    tf.reduce_mean(x, 1) ==> [1.,  2.]
    ```

    Args:
      input_tensor: The tensor to reduce. Should have numeric type.
      axis: The dimensions to reduce. If `None` (the default),
        reduces all dimensions.
      keep_dims: If true, retains reduced dimensions with length 1.
      name: A name for the operation (optional).
      reduction_indices: The old (deprecated) name for axis.

    Returns:
      The reduced tensor.
      help(tf.truncated_normal)
      Help on function truncated_normal in module tensorflow.python.ops.random_ops:
### truncated_normal
truncated_normal(shape, mean=0.0, stddev=1.0, dtype=tf.float32, seed=None, name=None)
输出截断的正态的分布随机值，通过指定正态分布的均值和标准差生成正态分布值。The generated values follow a normal distribution with specified mean and standard deviation, except that values whose magnitude is more than 2 standard deviations from the mean are dropped and re-picked.
Args:
- shape ：A以为整数tensor或者Python数组。
- stddev，mean:A0维Tensor或者Python值，数据类型为dtype指定的类型， A 0-D Tensor or Python value of type `dtype`. The mean of the
- seed: 一个Python整数，用于创建正态分布的种子。
              See
              [`set_random_seed`](../../api_docs/python/constant_op.md#set_random_seed)
              for behavior.
            name: A name for the operation (optional).

Returns:
一个被truncated的正态分布填满的tensor。
### conv2d
tensorflow.python.ops.gen_nn_ops:
conv2d(input, filter, strides, padding, use_cudnn_on_gpu=None, data_format=None, name=None)  
通过4维input和filter tensor计算一个2维卷积。
给定一个输入tensor的形状为[batch, in_height, in_width, in_channels]，filter或者kernel 的形状'[filter_height, filter_width, in_channels, out_channels]',操作按照下面执行：
1. Flattens一个Filter为2为矩阵，形状为`[filter_height * filter_width * in_channels, output_channels]`
2. 从输入tensor提取图片的patchs，到一个*virtual* tensor,它的形状为`[batch, out_height, out_width,filter_height * filter_width * in_channels]`
3. 对每个patch，右乘filter矩阵filter矩阵和图像image patch向量。
       `output[b, i, j, k] =
            sum_{di, dj, q} input[b, strides[1] * i + di, strides[2] * j + dj, q] * filter[di, dj, q, k]``

必须有`strides[0] = strides[3] = 1`对于通常情况下相同的由相同的水平或者竖直stride,`strides = [1, stride, stride, 1]`  
Args:  
- input: 一个`Tensor`.必须是下面的数据类型: `half`, `float32`, `float64`.
- filter: 一个`Tensor`. 必须由和`input`相同的数据类型。
- strides: 一个整形列表，一维，长度为4,滑窗的stride对于'input'的每一维，必须有和指定格式相同的维度。
- padding:一个字符串，可以为"SAME","VALID".padding的算法。
- use_cudnn_on_gpu:一个boolean值，默认为True。
- data_format:字符串， `"NHWC", "NCHW"`. 默认为`"NHWC"`。指定输入或者输出的数据格式，默认`"NHWC"`，按照如下顺序存储[batch, in_height, in_width, in_channels]，也可以按照下面的格式存储"NCHW"，存储顺序为[batch, in_channels, in_height, in_width]
- name:一个操作的名字。
- Returns:一个`Tensor`和`input`由相同的类型
### max_pool
Help on function max_pool in module tensorflow.python.ops.nn_ops: max_pool(value, ksize, strides, padding, data_format='NHWC', name=None)
     Performs the max pooling on the input.

     Args:
     value：一个4维`Tensor`，形状为 `[batch, height, width, channels]`,数据类型为`tf.float32`
     value: A 4-D `Tensor` with shape `[batch, height, width, channels]` and type `tf.float32`.
     ksize:一个整数列表，长度>=4.输入Tensor每一维Windows的大小
     ksize: A list of ints that has length >= 4.  The size of the window for each dimension of the input tensor.
     strides:长度>=4的整数列表，每一个输入tensor的滑动窗的stride。
     strides: A list of ints that has length >= 4.  The stride of the sliding window for each dimension of the input tensor.
     padding:一个字符串，`'VALID'` or `'SAME'`,padding算法。
     padding: A string, either `'VALID'` or `'SAME'`. The padding algorithm.
         See the @{tf.nn.convolution$comment here}
     data_format:一个字符串'NHWC'或者'NCHW'
    data_format: A string. 'NHWC' and 'NCHW' are supported.
     name:操作名字。
       name: Optional name for the operation.

     Returns:
     返回一个`tf.float32`的`Tensor`,,最大池化的输出结果。
       A `Tensor` with type `tf.float32`.  The max pooled output tensor.
## summary下面函数
- audio:
Help on function audio in module tensorflow.python.summary.summary:

audio(name, tensor, sample_rate, max_outputs=3, collections=None)
    Outputs a `Summary` protocol buffer with audio.
    
    The summary has up to `max_outputs` summary values containing audio. The
    audio is built from `tensor` which must be 3-D with shape `[batch_size,
    frames, channels]` or 2-D with shape `[batch_size, frames]`. The values are
    assumed to be in the range of `[-1.0, 1.0]` with a sample rate of
    `sample_rate`.
    
    The `tag` in the outputted Summary.Value protobufs is generated based on the
    name, with a suffix depending on the max_outputs setting:
    
    *  If `max_outputs` is 1, the summary value tag is '*name*/audio'.
    *  If `max_outputs` is greater than 1, the summary value tags are
       generated sequentially as '*name*/audio/0', '*name*/audio/1', etc
    
    Args:
      name: A name for the generated node. Will also serve as a series name in
        TensorBoard.
      tensor: A 3-D `float32` `Tensor` of shape `[batch_size, frames, channels]`
        or a 2-D `float32` `Tensor` of shape `[batch_size, frames]`.
      sample_rate: A Scalar `float32` `Tensor` indicating the sample rate of the
        signal in hertz.
      max_outputs: Max number of batch elements to generate audio for.
      collections: Optional list of ops.GraphKeys.  The collections to add the
        summary to.  Defaults to [_ops.GraphKeys.SUMMARIES]
    
    Returns:
      A scalar `Tensor` of type `string`. The serialized `Summary` protocol
      buffer.
- Event
Help on class Event in module tensorflow.core.util.event_pb2:

class Event(google.protobuf.message.Message)
 |  Abstract base class for protocol messages.
 |  
 |  Protocol message classes are almost always generated by the protocol
 |  compiler.  These generated types subclass Message and implement the methods
 |  shown below.
 |  
 |  TODO(robinson): Link to an HTML document here.
 |  
 |  TODO(robinson): Document that instances of this class will also
 |  have an Extensions attribute with __getitem__ and __setitem__.
 |  Again, not sure how to best convey this.
 |  
 |  TODO(robinson): Document that the class must also have a static
 |    RegisterExtension(extension_field) method.
 |    Not sure how to best express at this point.
 |  
 |  Method resolution order:
 |      Event
 |      google.protobuf.message.Message
 |      builtins.object
 |  
 |  Methods defined here:
 |  
 |  ByteSize(self)
 |  
 |  Clear = _Clear(self)
 |  
 |  ClearField(self, field_name)
 |  
 |  DiscardUnknownFields = _DiscardUnknownFields(self)
 |  
 |  FindInitializationErrors(self)
 |      Finds required fields which are not initialized.
 |      
 |      Returns:
 |        A list of strings.  Each string is a path to an uninitialized field from
 |        the top-level message, e.g. "foo.bar[5].baz".
 |  
 |  HasField(self, field_name)
 |  
 |  IsInitialized(self, errors=None)
 |      Checks if all required fields of a message are set.
 |      
 |      Args:
 |        errors:  A list which, if provided, will be populated with the field
 |                 paths of all missing required fields.
 |      
 |      Returns:
 |        True iff the specified message has all required fields set.
 |  
 |  ListFields(self)
 |  
 |  MergeFrom(self, msg)
 |  
 |  MergeFromString(self, serialized)
 |  
 |  SerializePartialToString(self)
 |  
 |  SerializeToString(self)
 |  
 |  SetInParent = Modified(self)
 |      Sets the _cached_byte_size_dirty bit to true,
 |      and propagates this to our listener iff this was a state change.
 |  
 |  WhichOneof(self, oneof_name)
 |      Returns the name of the currently set field inside a oneof, or None.
 |  
 |  _InternalParse = InternalParse(self, buffer, pos, end)
 |  
 |  _InternalSerialize = InternalSerialize(self, write_bytes)
 |  
 |  _Modified = Modified(self)
 |      Sets the _cached_byte_size_dirty bit to true,
 |      and propagates this to our listener iff this was a state change.
 |  
 |  _SetListener(self, listener)
 |  
 |  _UpdateOneofState(self, field)
 |      Sets field as the active field in its containing oneof.
 |      
 |      Will also delete currently active field in the oneof, if it is different
 |      from the argument. Does not mark the message as modified.
 |  
 |  __eq__(self, other)
 |  
 |  __init__ = init(self, **kwargs)
 |  
 |  __reduce__(self)
 |  
 |  __repr__(self)
 |  
 |  __str__(self)
 |  
 |  __unicode__(self)
 |  
 |  ----------------------------------------------------------------------
 |  Static methods defined here:
 |  
 |  FromString(s)
 |  
 |  RegisterExtension(extension_handle)
 |      # TODO(robinson): This probably needs to be thread-safe(?)
 |  
 |  ----------------------------------------------------------------------
 |  Data descriptors defined here:
 |  
 |  __weakref__
 |      list of weak references to the object (if defined)
 |  
 |  _cached_byte_size
 |  
 |  _cached_byte_size_dirty
 |  
 |  _fields
 |  
 |  _is_present_in_parent
 |  
 |  _listener
 |  
 |  _listener_for_children
 |  
 |  _oneofs
 |  
 |  _unknown_fields
 |  
 |  file_version
 |      Magic attribute generated for "file_version" proto field.
 |  
 |  graph_def
 |      Magic attribute generated for "graph_def" proto field.
 |  
 |  log_message
 |      Magic attribute generated for "log_message" proto field.
 |  
 |  meta_graph_def
 |      Magic attribute generated for "meta_graph_def" proto field.
 |  
 |  session_log
 |      Magic attribute generated for "session_log" proto field.
 |  
 |  step
 |      Magic attribute generated for "step" proto field.
 |  
 |  summary
 |      Magic attribute generated for "summary" proto field.
 |  
 |  tagged_run_metadata
 |      Magic attribute generated for "tagged_run_metadata" proto field.
 |  
 |  wall_time
 |      Magic attribute generated for "wall_time" proto field.
 |  
 |  ----------------------------------------------------------------------
 |  Data and other attributes defined here:
 |  
 |  DESCRIPTOR = <google.protobuf.descriptor.Descriptor object>
 |  
 |  FILE_VERSION_FIELD_NUMBER = 3
 |  
 |  GRAPH_DEF_FIELD_NUMBER = 4
 |  
 |  LOG_MESSAGE_FIELD_NUMBER = 6
 |  
 |  META_GRAPH_DEF_FIELD_NUMBER = 9
 |  
 |  SESSION_LOG_FIELD_NUMBER = 7
 |  
 |  STEP_FIELD_NUMBER = 2
 |  
 |  SUMMARY_FIELD_NUMBER = 5
 |  
 |  TAGGED_RUN_METADATA_FIELD_NUMBER = 8
 |  
 |  WALL_TIME_FIELD_NUMBER = 1
 |  
 |  _decoders_by_tag = {b'\t': (<function _SimpleDecoder.<locals>.Specific...
 |  
 |  _extensions_by_name = {}
 |  
 |  _extensions_by_number = {}
 |  
 |  ----------------------------------------------------------------------
 |  Methods inherited from google.protobuf.message.Message:
 |  
 |  ClearExtension(self, extension_handle)
 |  
 |  CopyFrom(self, other_msg)
 |      Copies the content of the specified message into the current message.
 |      
 |      The method clears the current message and then merges the specified
 |      message using MergeFrom.
 |      
 |      Args:
 |        other_msg: Message to copy into the current one.
 |  
 |  HasExtension(self, extension_handle)
 |  
 |  ParseFromString(self, serialized)
 |      Parse serialized protocol buffer data into this message.
 |      
 |      Like MergeFromString(), except we clear the object first and
 |      do not return the value that MergeFromString returns.
 |  
 |  __deepcopy__(self, memo=None)
 |  
 |  __getstate__(self)
 |      Support the pickle protocol.
 |  
 |  __hash__(self)
 |      Return hash(self).
 |  
 |  __ne__(self, other_msg)
 |      Return self!=value.
 |  
 |  __setstate__(self, state)
 |      Support the pickle protocol.
- FileWriter
Help on class FileWriter in module tensorflow.python.summary.writer.writer:

class FileWriter(SummaryToEventTransformer)
 |  Writes `Summary` protocol buffers to event files.
 |  
 |  The `FileWriter` class provides a mechanism to create an event file in a
 |  given directory and add summaries and events to it. The class updates the
 |  file contents asynchronously. This allows a training program to call methods
 |  to add data to the file directly from the training loop, without slowing down
 |  training.
 |  
 |  Method resolution order:
 |      FileWriter
 |      SummaryToEventTransformer
 |      builtins.object
 |  
 |  Methods defined here:
 |  
 |  __init__(self, logdir, graph=None, max_queue=10, flush_secs=120, graph_def=None)
 |      Creates a `FileWriter` and an event file.
 |      
 |      On construction the summary writer creates a new event file in `logdir`.
 |      This event file will contain `Event` protocol buffers constructed when you
 |      call one of the following functions: `add_summary()`, `add_session_log()`,
 |      `add_event()`, or `add_graph()`.
 |      
 |      If you pass a `Graph` to the constructor it is added to
 |      the event file. (This is equivalent to calling `add_graph()` later).
 |      
 |      TensorBoard will pick the graph from the file and display it graphically so
 |      you can interactively explore the graph you built. You will usually pass
 |      the graph from the session in which you launched it:
 |      
 |      ```python
 |      ...create a graph...
 |      # Launch the graph in a session.
 |      sess = tf.Session()
 |      # Create a summary writer, add the 'graph' to the event file.
 |      writer = tf.summary.FileWriter(<some-directory>, sess.graph)
 |      ```
 |      
 |      The other arguments to the constructor control the asynchronous writes to
 |      the event file:
 |      
 |      *  `flush_secs`: How often, in seconds, to flush the added summaries
 |         and events to disk.
 |      *  `max_queue`: Maximum number of summaries or events pending to be
 |         written to disk before one of the 'add' calls block.
 |      
 |      Args:
 |        logdir: A string. Directory where event file will be written.
 |        graph: A `Graph` object, such as `sess.graph`.
 |        max_queue: Integer. Size of the queue for pending events and summaries.
 |        flush_secs: Number. How often, in seconds, to flush the
 |          pending events and summaries to disk.
 |        graph_def: DEPRECATED: Use the `graph` argument instead.
 |  
 |  add_event(self, event)
 |      Adds an event to the event file.
 |      
 |      Args:
 |        event: An `Event` protocol buffer.
 |  
 |  close(self)
 |      Flushes the event file to disk and close the file.
 |      
 |      Call this method when you do not need the summary writer anymore.
 |  
 |  flush(self)
 |      Flushes the event file to disk.
 |      
 |      Call this method to make sure that all pending events have been written to
 |      disk.
 |  
 |  get_logdir(self)
 |      Returns the directory where event file will be written.
 |  
 |  reopen(self)
 |      Reopens the EventFileWriter.
 |      
 |      Can be called after `close()` to add more events in the same directory.
 |      The events will go into a new events file.
 |      
 |      Does nothing if the EventFileWriter was not closed.
 |  
 |  ----------------------------------------------------------------------
 |  Methods inherited from SummaryToEventTransformer:
 |  
 |  add_graph(self, graph, global_step=None, graph_def=None)
 |      Adds a `Graph` to the event file.
 |      
 |      The graph described by the protocol buffer will be displayed by
 |      TensorBoard. Most users pass a graph in the constructor instead.
 |      
 |      Args:
 |        graph: A `Graph` object, such as `sess.graph`.
 |        global_step: Number. Optional global step counter to record with the
 |          graph.
 |        graph_def: DEPRECATED. Use the `graph` parameter instead.
 |      
 |      Raises:
 |        ValueError: If both graph and graph_def are passed to the method.
 |  
 |  add_meta_graph(self, meta_graph_def, global_step=None)
 |      Adds a `MetaGraphDef` to the event file.
 |      
 |      The `MetaGraphDef` allows running the given graph via
 |      `saver.import_meta_graph()`.
 |      
 |      Args:
 |        meta_graph_def: A `MetaGraphDef` object, often as retured by
 |          `saver.export_meta_graph()`.
 |        global_step: Number. Optional global step counter to record with the
 |          graph.
 |      
 |      Raises:
 |        TypeError: If both `meta_graph_def` is not an instance of `MetaGraphDef`.
 |  
 |  add_run_metadata(self, run_metadata, tag, global_step=None)
 |      Adds a metadata information for a single session.run() call.
 |      
 |      Args:
 |        run_metadata: A `RunMetadata` protobuf object.
 |        tag: The tag name for this metadata.
 |        global_step: Number. Optional global step counter to record with the
 |          StepStats.
 |      
 |      Raises:
 |        ValueError: If the provided tag was already used for this type of event.
 |  
 |  add_session_log(self, session_log, global_step=None)
 |      Adds a `SessionLog` protocol buffer to the event file.
 |      
 |      This method wraps the provided session in an `Event` protocol buffer
 |      and adds it to the event file.
 |      
 |      Args:
 |        session_log: A `SessionLog` protocol buffer.
 |        global_step: Number. Optional global step value to record with the
 |          summary.
 |  
 |  add_summary(self, summary, global_step=None)
 |      Adds a `Summary` protocol buffer to the event file.
 |      
 |      This method wraps the provided summary in an `Event` protocol buffer
 |      and adds it to the event file.
 |      
 |      You can pass the result of evaluating any summary op, using
 |      @{tf.Session.run} or
 |      @{tf.Tensor.eval}, to this
 |      function. Alternatively, you can pass a `tf.Summary` protocol
 |      buffer that you populate with your own data. The latter is
 |      commonly done to report evaluation results in event files.
 |      
 |      Args:
 |        summary: A `Summary` protocol buffer, optionally serialized as a string.
 |        global_step: Number. Optional global step value to record with the
 |          summary.
 |  
 |  ----------------------------------------------------------------------
 |  Data descriptors inherited from SummaryToEventTransformer:
 |  
 |  __dict__
 |      dictionary for instance variables (if defined)
 |  
 |  __weakref__
 |      list of weak references to the object (if defined)
- FileWriterCache

- get_summary_description
Help on class FileWriter in module tensorflow.python.summary.writer.writer:

class FileWriter(SummaryToEventTransformer)
 |  Writes `Summary` protocol buffers to event files.
 |  
 |  The `FileWriter` class provides a mechanism to create an event file in a
 |  given directory and add summaries and events to it. The class updates the
 |  file contents asynchronously. This allows a training program to call methods
 |  to add data to the file directly from the training loop, without slowing down
 |  training.
 |  
 |  Method resolution order:
 |      FileWriter
 |      SummaryToEventTransformer
 |      builtins.object
 |  
 |  Methods defined here:
 |  
 |  __init__(self, logdir, graph=None, max_queue=10, flush_secs=120, graph_def=None)
 |      Creates a `FileWriter` and an event file.
 |      
 |      On construction the summary writer creates a new event file in `logdir`.
 |      This event file will contain `Event` protocol buffers constructed when you
 |      call one of the following functions: `add_summary()`, `add_session_log()`,
 |      `add_event()`, or `add_graph()`.
 |      
 |      If you pass a `Graph` to the constructor it is added to
 |      the event file. (This is equivalent to calling `add_graph()` later).
 |      
 |      TensorBoard will pick the graph from the file and display it graphically so
 |      you can interactively explore the graph you built. You will usually pass
 |      the graph from the session in which you launched it:
 |      
 |      ```python
 |      ...create a graph...
 |      # Launch the graph in a session.
 |      sess = tf.Session()
 |      # Create a summary writer, add the 'graph' to the event file.
 |      writer = tf.summary.FileWriter(<some-directory>, sess.graph)
 |      ```
 |      
 |      The other arguments to the constructor control the asynchronous writes to
 |      the event file:
 |      
 |      *  `flush_secs`: How often, in seconds, to flush the added summaries
 |         and events to disk.
 |      *  `max_queue`: Maximum number of summaries or events pending to be
 |         written to disk before one of the 'add' calls block.
 |      
 |      Args:
 |        logdir: A string. Directory where event file will be written.
 |        graph: A `Graph` object, such as `sess.graph`.
 |        max_queue: Integer. Size of the queue for pending events and summaries.
 |        flush_secs: Number. How often, in seconds, to flush the
 |          pending events and summaries to disk.
 |        graph_def: DEPRECATED: Use the `graph` argument instead.
 |  
 |  add_event(self, event)
 |      Adds an event to the event file.
 |      
 |      Args:
 |        event: An `Event` protocol buffer.
 |  
 |  close(self)
 |      Flushes the event file to disk and close the file.
 |      
 |      Call this method when you do not need the summary writer anymore.
 |  
 |  flush(self)
 |      Flushes the event file to disk.
 |      
 |      Call this method to make sure that all pending events have been written to
 |      disk.
 |  
 |  get_logdir(self)
 |      Returns the directory where event file will be written.
 |  
 |  reopen(self)
 |      Reopens the EventFileWriter.
 |      
 |      Can be called after `close()` to add more events in the same directory.
 |      The events will go into a new events file.
 |      
 |      Does nothing if the EventFileWriter was not closed.
 |  
 |  ----------------------------------------------------------------------
 |  Methods inherited from SummaryToEventTransformer:
 |  
 |  add_graph(self, graph, global_step=None, graph_def=None)
 |      Adds a `Graph` to the event file.
 |      
 |      The graph described by the protocol buffer will be displayed by
 |      TensorBoard. Most users pass a graph in the constructor instead.
 |      
 |      Args:
 |        graph: A `Graph` object, such as `sess.graph`.
 |        global_step: Number. Optional global step counter to record with the
 |          graph.
 |        graph_def: DEPRECATED. Use the `graph` parameter instead.
 |      
 |      Raises:
 |        ValueError: If both graph and graph_def are passed to the method.
 |  
 |  add_meta_graph(self, meta_graph_def, global_step=None)
 |      Adds a `MetaGraphDef` to the event file.
 |      
 |      The `MetaGraphDef` allows running the given graph via
 |      `saver.import_meta_graph()`.
 |      
 |      Args:
 |        meta_graph_def: A `MetaGraphDef` object, often as retured by
 |          `saver.export_meta_graph()`.
 |        global_step: Number. Optional global step counter to record with the
 |          graph.
 |      
 |      Raises:
 |        TypeError: If both `meta_graph_def` is not an instance of `MetaGraphDef`.
 |  
 |  add_run_metadata(self, run_metadata, tag, global_step=None)
 |      Adds a metadata information for a single session.run() call.
 |      
 |      Args:
 |        run_metadata: A `RunMetadata` protobuf object.
 |        tag: The tag name for this metadata.
 |        global_step: Number. Optional global step counter to record with the
 |          StepStats.
 |      
 |      Raises:
 |        ValueError: If the provided tag was already used for this type of event.
 |  
 |  add_session_log(self, session_log, global_step=None)
 |      Adds a `SessionLog` protocol buffer to the event file.
 |      
 |      This method wraps the provided session in an `Event` protocol buffer
 |      and adds it to the event file.
 |      
 |      Args:
 |        session_log: A `SessionLog` protocol buffer.
 |        global_step: Number. Optional global step value to record with the
 |          summary.
 |  
 |  add_summary(self, summary, global_step=None)
 |      Adds a `Summary` protocol buffer to the event file.
 |      
 |      This method wraps the provided summary in an `Event` protocol buffer
 |      and adds it to the event file.
 |      
 |      You can pass the result of evaluating any summary op, using
 |      @{tf.Session.run} or
 |      @{tf.Tensor.eval}, to this
 |      function. Alternatively, you can pass a `tf.Summary` protocol
 |      buffer that you populate with your own data. The latter is
 |      commonly done to report evaluation results in event files.
 |      
 |      Args:
 |        summary: A `Summary` protocol buffer, optionally serialized as a string.
 |        global_step: Number. Optional global step value to record with the
 |          summary.
 |  
 |  ----------------------------------------------------------------------
 |  Data descriptors inherited from SummaryToEventTransformer:
 |  
 |  __dict__
 |      dictionary for instance variables (if defined)
 |  
 |  __weakref__
 |      list of weak references to the object (if defined)


help(tf.summary.FileWriterCache)
Help on class FileWriterCache in module tensorflow.python.summary.writer.writer_cache:

class FileWriterCache(builtins.object)
 |  Cache for file writers.
 |  
 |  This class caches file writers, one per directory.
 |  
 |  Static methods defined here:
 |  
 |  clear()
 |      Clear cached summary writers. Currently only used for unit tests.
 |  
 |  get(logdir)
 |      Returns the FileWriter for the specified directory.
 |      
 |      Args:
 |        logdir: str, name of the directory.
 |      
 |      Returns:
 |        A `FileWriter`.
 |  
 |  ----------------------------------------------------------------------
 |  Data descriptors defined here:
 |  
 |  __dict__
 |      dictionary for instance variables (if defined)
 |  
 |  __weakref__
 |      list of weak references to the object (if defined)
- histgram
Help on class FileWriter in module tensorflow.python.summary.writer.writer:

class FileWriter(SummaryToEventTransformer)
 |  Writes `Summary` protocol buffers to event files.
 |  
 |  The `FileWriter` class provides a mechanism to create an event file in a
 |  given directory and add summaries and events to it. The class updates the
 |  file contents asynchronously. This allows a training program to call methods
 |  to add data to the file directly from the training loop, without slowing down
 |  training.
 |  
 |  Method resolution order:
 |      FileWriter
 |      SummaryToEventTransformer
 |      builtins.object
 |  
 |  Methods defined here:
 |  
 |  __init__(self, logdir, graph=None, max_queue=10, flush_secs=120, graph_def=None)
 |      Creates a `FileWriter` and an event file.
 |      
 |      On construction the summary writer creates a new event file in `logdir`.
 |      This event file will contain `Event` protocol buffers constructed when you
 |      call one of the following functions: `add_summary()`, `add_session_log()`,
 |      `add_event()`, or `add_graph()`.
 |      
 |      If you pass a `Graph` to the constructor it is added to
 |      the event file. (This is equivalent to calling `add_graph()` later).
 |      
 |      TensorBoard will pick the graph from the file and display it graphically so
 |      you can interactively explore the graph you built. You will usually pass
 |      the graph from the session in which you launched it:
 |      
 |      ```python
 |      ...create a graph...
 |      # Launch the graph in a session.
 |      sess = tf.Session()
 |      # Create a summary writer, add the 'graph' to the event file.
 |      writer = tf.summary.FileWriter(<some-directory>, sess.graph)
 |      ```
 |      
 |      The other arguments to the constructor control the asynchronous writes to
 |      the event file:
 |      
 |      *  `flush_secs`: How often, in seconds, to flush the added summaries
 |         and events to disk.
 |      *  `max_queue`: Maximum number of summaries or events pending to be
 |         written to disk before one of the 'add' calls block.
 |      
 |      Args:
 |        logdir: A string. Directory where event file will be written.
 |        graph: A `Graph` object, such as `sess.graph`.
 |        max_queue: Integer. Size of the queue for pending events and summaries.
 |        flush_secs: Number. How often, in seconds, to flush the
 |          pending events and summaries to disk.
 |        graph_def: DEPRECATED: Use the `graph` argument instead.
 |  
 |  add_event(self, event)
 |      Adds an event to the event file.
 |      
 |      Args:
 |        event: An `Event` protocol buffer.
 |  
 |  close(self)
 |      Flushes the event file to disk and close the file.
 |      
 |      Call this method when you do not need the summary writer anymore.
 |  
 |  flush(self)
 |      Flushes the event file to disk.
 |      
 |      Call this method to make sure that all pending events have been written to
 |      disk.
 |  
 |  get_logdir(self)
 |      Returns the directory where event file will be written.
 |  
 |  reopen(self)
 |      Reopens the EventFileWriter.
 |      
 |      Can be called after `close()` to add more events in the same directory.
 |      The events will go into a new events file.
 |      
 |      Does nothing if the EventFileWriter was not closed.
 |  
 |  ----------------------------------------------------------------------
 |  Methods inherited from SummaryToEventTransformer:
 |  
 |  add_graph(self, graph, global_step=None, graph_def=None)
 |      Adds a `Graph` to the event file.
 |      
 |      The graph described by the protocol buffer will be displayed by
 |      TensorBoard. Most users pass a graph in the constructor instead.
 |      
 |      Args:
 |        graph: A `Graph` object, such as `sess.graph`.
 |        global_step: Number. Optional global step counter to record with the
 |          graph.
 |        graph_def: DEPRECATED. Use the `graph` parameter instead.
 |      
 |      Raises:
 |        ValueError: If both graph and graph_def are passed to the method.
 |  
 |  add_meta_graph(self, meta_graph_def, global_step=None)
 |      Adds a `MetaGraphDef` to the event file.
 |      
 |      The `MetaGraphDef` allows running the given graph via
 |      `saver.import_meta_graph()`.
 |      
 |      Args:
 |        meta_graph_def: A `MetaGraphDef` object, often as retured by
 |          `saver.export_meta_graph()`.
 |        global_step: Number. Optional global step counter to record with the
 |          graph.
 |      
 |      Raises:
 |        TypeError: If both `meta_graph_def` is not an instance of `MetaGraphDef`.
 |  
 |  add_run_metadata(self, run_metadata, tag, global_step=None)
 |      Adds a metadata information for a single session.run() call.
 |      
 |      Args:
 |        run_metadata: A `RunMetadata` protobuf object.
 |        tag: The tag name for this metadata.
 |        global_step: Number. Optional global step counter to record with the
 |          StepStats.
 |      
 |      Raises:
 |        ValueError: If the provided tag was already used for this type of event.
 |  
 |  add_session_log(self, session_log, global_step=None)
 |      Adds a `SessionLog` protocol buffer to the event file.
 |      
 |      This method wraps the provided session in an `Event` protocol buffer
 |      and adds it to the event file.
 |      
 |      Args:
 |        session_log: A `SessionLog` protocol buffer.
 |        global_step: Number. Optional global step value to record with the
 |          summary.
 |  
 |  add_summary(self, summary, global_step=None)
 |      Adds a `Summary` protocol buffer to the event file.
 |      
 |      This method wraps the provided summary in an `Event` protocol buffer
 |      and adds it to the event file.
 |      
 |      You can pass the result of evaluating any summary op, using
 |      @{tf.Session.run} or
 |      @{tf.Tensor.eval}, to this
 |      function. Alternatively, you can pass a `tf.Summary` protocol
 |      buffer that you populate with your own data. The latter is
 |      commonly done to report evaluation results in event files.
 |      
 |      Args:
 |        summary: A `Summary` protocol buffer, optionally serialized as a string.
 |        global_step: Number. Optional global step value to record with the
 |          summary.
 |  
 |  ----------------------------------------------------------------------
 |  Data descriptors inherited from SummaryToEventTransformer:
 |  
 |  __dict__
 |      dictionary for instance variables (if defined)
 |  
 |  __weakref__
 |      list of weak references to the object (if defined)


help(tf.summary.FileWriterCache)
Help on class FileWriterCache in module tensorflow.python.summary.writer.writer_cache:

class FileWriterCache(builtins.object)
 |  Cache for file writers.
 |  
 |  This class caches file writers, one per directory.
 |  
 |  Static methods defined here:
 |  
 |  clear()
 |      Clear cached summary writers. Currently only used for unit tests.
 |  
 |  get(logdir)
 |      Returns the FileWriter for the specified directory.
 |      
 |      Args:
 |        logdir: str, name of the directory.
 |      
 |      Returns:
 |        A `FileWriter`.
 |  
 |  ----------------------------------------------------------------------
 |  Data descriptors defined here:
 |  
 |  __dict__
 |      dictionary for instance variables (if defined)
 |  
 |  __weakref__
 |      list of weak references to the object (if defined)


help(tf.summary.histogram)
Help on function histogram in module tensorflow.python.summary.summary:

histogram(name, values, collections=None)
    Outputs a `Summary` protocol buffer with a histogram.
    
    The generated
    [`Summary`](https://www.tensorflow.org/code/tensorflow/core/framework/summary.proto)
    has one summary value containing a histogram for `values`.
    
    This op reports an `InvalidArgument` error if any value is not finite.
    
    Args:
      name: A name for the generated node. Will also serve as a series name in
        TensorBoard.
      values: A real numeric `Tensor`. Any shape. Values to use to
        build the histogram.
      collections: Optional list of graph collections keys. The new summary op is
        added to these collections. Defaults to `[GraphKeys.SUMMARIES]`.
    
    Returns:
      A scalar `Tensor` of type `string`. The serialized `Summary` protocol
      buffer.
- image
 Help on function image in module tensorflow.python.summary.summary:

image(name, tensor, max_outputs=3, collections=None)
    Outputs a `Summary` protocol buffer with images.
    
    The summary has up to `max_outputs` summary values containing images. The
    images are built from `tensor` which must be 4-D with shape `[batch_size,
    height, width, channels]` and where `channels` can be:
    
    *  1: `tensor` is interpreted as Grayscale.
    *  3: `tensor` is interpreted as RGB.
    *  4: `tensor` is interpreted as RGBA.
    
    The images have the same number of channels as the input tensor. For float
    input, the values are normalized one image at a time to fit in the range
    `[0, 255]`.  `uint8` values are unchanged.  The op uses two different
    normalization algorithms:
    
    *  If the input values are all positive, they are rescaled so the largest one
       is 255.
    
    *  If any input value is negative, the values are shifted so input value 0.0
       is at 127.  They are then rescaled so that either the smallest value is 0,
       or the largest one is 255.
    
    The `tag` in the outputted Summary.Value protobufs is generated based on the
    name, with a suffix depending on the max_outputs setting:
    
    *  If `max_outputs` is 1, the summary value tag is '*name*/image'.
    *  If `max_outputs` is greater than 1, the summary value tags are
       generated sequentially as '*name*/image/0', '*name*/image/1', etc.
    
    Args:
      name: A name for the generated node. Will also serve as a series name in
        TensorBoard.
      tensor: A 4-D `uint8` or `float32` `Tensor` of shape `[batch_size, height,
        width, channels]` where `channels` is 1, 3, or 4.
      max_outputs: Max number of batch elements to generate images for.
      collections: Optional list of ops.GraphKeys.  The collections to add the
        summary to.  Defaults to [_ops.GraphKeys.SUMMARIES]
    
    Returns:
      A scalar `Tensor` of type `string`. The serialized `Summary` protocol
      buffer.
- merge
Help on function merge in module tensorflow.python.summary.summary:

merge(inputs, collections=None, name=None)
    Merges summaries.
    
    This op creates a
    [`Summary`](https://www.tensorflow.org/code/tensorflow/core/framework/summary.proto)
    protocol buffer that contains the union of all the values in the input
    summaries.
- merge_all
Help on function merge_all in module tensorflow.python.summary.summary:

merge_all(key='summaries')
    Merges all summaries collected in the default graph.
    
    Args:
      key: `GraphKey` used to collect the summaries.  Defaults to
        `GraphKeys.SUMMARIES`.
    
    Returns:
      If no summaries were collected, returns None.  Otherwise returns a scalar
      `Tensor` of type `string` containing the serialized `Summary` protocol
      buffer resulting from the merging.
- scalar
Help on function scalar in module tensorflow.python.summary.summary:

scalar(name, tensor, collections=None)
    Outputs a `Summary` protocol buffer containing a single scalar value.
    
    The generated Summary has a Tensor.proto containing the input Tensor.
    
    Args:
      name: A name for the generated node. Will also serve as the series name in
        TensorBoard.
      tensor: A real numeric Tensor containing a single value.
      collections: Optional list of graph collections keys. The new summary op is
        added to these collections. Defaults to `[GraphKeys.SUMMARIES]`.
    
    Returns:
      A scalar `Tensor` of type `string`. Which contains a `Summary` protobuf.
    
    Raises:
      ValueError: If tensor has the wrong shape or type.
- SessionLog
Help on class SessionLog in module tensorflow.core.util.event_pb2:

class SessionLog(google.protobuf.message.Message)
 |  Abstract base class for protocol messages.
 |  
 |  Protocol message classes are almost always generated by the protocol
 |  compiler.  These generated types subclass Message and implement the methods
 |  shown below.
 |  
 |  TODO(robinson): Link to an HTML document here.
 |  
 |  TODO(robinson): Document that instances of this class will also
 |  have an Extensions attribute with __getitem__ and __setitem__.
 |  Again, not sure how to best convey this.
 |  
 |  TODO(robinson): Document that the class must also have a static
 |    RegisterExtension(extension_field) method.
 |    Not sure how to best express at this point.
 |  
 |  Method resolution order:
 |      SessionLog
 |      google.protobuf.message.Message
 |      builtins.object
 |  
 |  Methods defined here:
 |  
 |  ByteSize(self)
 |  
 |  Clear = _Clear(self)
 |  
 |  ClearField(self, field_name)
 |  
 |  DiscardUnknownFields = _DiscardUnknownFields(self)
 |  
 |  FindInitializationErrors(self)
 |      Finds required fields which are not initialized.
 |      
 |      Returns:
 |        A list of strings.  Each string is a path to an uninitialized field from
 |        the top-level message, e.g. "foo.bar[5].baz".
 |  
 |  HasField(self, field_name)
 |  
 |  IsInitialized(self, errors=None)
 |      Checks if all required fields of a message are set.
 |      
 |      Args:
 |        errors:  A list which, if provided, will be populated with the field
 |                 paths of all missing required fields.
 |      
 |      Returns:
 |        True iff the specified message has all required fields set.
 |  
 |  ListFields(self)
 |  
 |  MergeFrom(self, msg)
 |  
 |  MergeFromString(self, serialized)
 |  
 |  SerializePartialToString(self)
 |  
 |  SerializeToString(self)
 |  
 |  SetInParent = Modified(self)
 |      Sets the _cached_byte_size_dirty bit to true,
 |      and propagates this to our listener iff this was a state change.
 |  
 |  WhichOneof(self, oneof_name)
 |      Returns the name of the currently set field inside a oneof, or None.
 |  
 |  _InternalParse = InternalParse(self, buffer, pos, end)
 |  
 |  _InternalSerialize = InternalSerialize(self, write_bytes)
 |  
 |  _Modified = Modified(self)
 |      Sets the _cached_byte_size_dirty bit to true,
 |      and propagates this to our listener iff this was a state change.
 |  
 |  _SetListener(self, listener)
 |  
 |  _UpdateOneofState(self, field)
 |      Sets field as the active field in its containing oneof.
 |      
 |      Will also delete currently active field in the oneof, if it is different
 |      from the argument. Does not mark the message as modified.
 |  
 |  __eq__(self, other)
 |  
 |  __init__ = init(self, **kwargs)
 |  
 |  __reduce__(self)
 |  
 |  __repr__(self)
 |  
 |  __str__(self)
 |  
 |  __unicode__(self)
 |  
 |  ----------------------------------------------------------------------
 |  Static methods defined here:
 |  
 |  FromString(s)
 |  
 |  RegisterExtension(extension_handle)
 |      # TODO(robinson): This probably needs to be thread-safe(?)
 |  
 |  ----------------------------------------------------------------------
 |  Data descriptors defined here:
 |  
 |  __weakref__
 |      list of weak references to the object (if defined)
 |  
 |  _cached_byte_size
 |  
 |  _cached_byte_size_dirty
 |  
 |  _fields
 |  
 |  _is_present_in_parent
 |  
 |  _listener
 |  
 |  _listener_for_children
 |  
 |  _oneofs
 |  
 |  _unknown_fields
 |  
 |  checkpoint_path
 |      Magic attribute generated for "checkpoint_path" proto field.
 |  
 |  msg
 |      Magic attribute generated for "msg" proto field.
 |  
 |  status
 |      Magic attribute generated for "status" proto field.
 |  
 |  ----------------------------------------------------------------------
 |  Data and other attributes defined here:
 |  
 |  CHECKPOINT = 3
 |  
 |  CHECKPOINT_PATH_FIELD_NUMBER = 2
 |  
 |  DESCRIPTOR = <google.protobuf.descriptor.Descriptor object>
 |  
 |  MSG_FIELD_NUMBER = 3
 |  
 |  START = 1
 |  
 |  STATUS_FIELD_NUMBER = 1
 |  
 |  STATUS_UNSPECIFIED = 0
 |  
 |  STOP = 2
 |  
 |  SessionStatus = <google.protobuf.internal.enum_type_wrapper.EnumTypeWr...
 |  
 |  _decoders_by_tag = {b'\x08': (<function _SimpleDecoder.<locals>.Specif...
 |  
 |  _extensions_by_name = {}
 |  
 |  _extensions_by_number = {}
 |  
 |  ----------------------------------------------------------------------
 |  Methods inherited from google.protobuf.message.Message:
 |  
 |  ClearExtension(self, extension_handle)
 |  
 |  CopyFrom(self, other_msg)
 |      Copies the content of the specified message into the current message.
 |      
 |      The method clears the current message and then merges the specified
 |      message using MergeFrom.
 |      
 |      Args:
 |        other_msg: Message to copy into the current one.
 |  
 |  HasExtension(self, extension_handle)
 |  
 |  ParseFromString(self, serialized)
 |      Parse serialized protocol buffer data into this message.
 |      
 |      Like MergeFromString(), except we clear the object first and
 |      do not return the value that MergeFromString returns.
 |  
 |  __deepcopy__(self, memo=None)
 |  
 |  __getstate__(self)
 |      Support the pickle protocol.
 |  
 |  __hash__(self)
 |      Return hash(self).
 |  
 |  __ne__(self, other_msg)
 |      Return self!=value.
 |  
 |  __setstate__(self, state)
 |      Support the pickle protocol.
- Summary
Help on class Summary in module tensorflow.core.framework.summary_pb2:

class Summary(google.protobuf.message.Message)
 |  Abstract base class for protocol messages.
 |  
 |  Protocol message classes are almost always generated by the protocol
 |  compiler.  These generated types subclass Message and implement the methods
 |  shown below.
 |  
 |  TODO(robinson): Link to an HTML document here.
 |  
 |  TODO(robinson): Document that instances of this class will also
 |  have an Extensions attribute with __getitem__ and __setitem__.
 |  Again, not sure how to best convey this.
 |  
 |  TODO(robinson): Document that the class must also have a static
 |    RegisterExtension(extension_field) method.
 |    Not sure how to best express at this point.
 |  
 |  Method resolution order:
 |      Summary
 |      google.protobuf.message.Message
 |      builtins.object
 |  
 |  Methods defined here:
 |  
 |  ByteSize(self)
 |  
 |  Clear = _Clear(self)
 |  
 |  ClearField(self, field_name)
 |  
 |  DiscardUnknownFields = _DiscardUnknownFields(self)
 |  
 |  FindInitializationErrors(self)
 |      Finds required fields which are not initialized.
 |      
 |      Returns:
 |        A list of strings.  Each string is a path to an uninitialized field from
 |        the top-level message, e.g. "foo.bar[5].baz".
 |  
 |  HasField(self, field_name)
 |  
 |  IsInitialized(self, errors=None)
 |      Checks if all required fields of a message are set.
 |      
 |      Args:
 |        errors:  A list which, if provided, will be populated with the field
 |                 paths of all missing required fields.
 |      
 |      Returns:
 |        True iff the specified message has all required fields set.
 |  
 |  ListFields(self)
 |  
 |  MergeFrom(self, msg)
 |  
 |  MergeFromString(self, serialized)
 |  
 |  SerializePartialToString(self)
 |  
 |  SerializeToString(self)
 |  
 |  SetInParent = Modified(self)
 |      Sets the _cached_byte_size_dirty bit to true,
 |      and propagates this to our listener iff this was a state change.
 |  
 |  WhichOneof(self, oneof_name)
 |      Returns the name of the currently set field inside a oneof, or None.
 |  
 |  _InternalParse = InternalParse(self, buffer, pos, end)
 |  
 |  _InternalSerialize = InternalSerialize(self, write_bytes)
 |  
 |  _Modified = Modified(self)
 |      Sets the _cached_byte_size_dirty bit to true,
 |      and propagates this to our listener iff this was a state change.
 |  
 |  _SetListener(self, listener)
 |  
 |  _UpdateOneofState(self, field)
 |      Sets field as the active field in its containing oneof.
 |      
 |      Will also delete currently active field in the oneof, if it is different
 |      from the argument. Does not mark the message as modified.
 |  
 |  __eq__(self, other)
 |  
 |  __init__ = init(self, **kwargs)
 |  
 |  __reduce__(self)
 |  
 |  __repr__(self)
 |  
 |  __str__(self)
 |  
 |  __unicode__(self)
 |  
 |  ----------------------------------------------------------------------
 |  Static methods defined here:
 |  
 |  FromString(s)
 |  
 |  RegisterExtension(extension_handle)
 |      # TODO(robinson): This probably needs to be thread-safe(?)
 |  
 |  ----------------------------------------------------------------------
 |  Data descriptors defined here:
 |  
 |  __weakref__
 |      list of weak references to the object (if defined)
 |  
 |  _cached_byte_size
 |  
 |  _cached_byte_size_dirty
 |  
 |  _fields
 |  
 |  _is_present_in_parent
 |  
 |  _listener
 |  
 |  _listener_for_children
 |  
 |  _oneofs
 |  
 |  _unknown_fields
 |  
 |  value
 |      Magic attribute generated for "value" proto field.
 |  
 |  ----------------------------------------------------------------------
 |  Data and other attributes defined here:
 |  
 |  Audio = <class 'tensorflow.core.framework.summary_pb2.Audio'>
 |      Abstract base class for protocol messages.
 |      
 |      Protocol message classes are almost always generated by the protocol
 |      compiler.  These generated types subclass Message and implement the methods
 |      shown below.
 |      
 |      TODO(robinson): Link to an HTML document here.
 |      
 |      TODO(robinson): Document that instances of this class will also
 |      have an Extensions attribute with __getitem__ and __setitem__.
 |      Again, not sure how to best convey this.
 |      
 |      TODO(robinson): Document that the class must also have a static
 |        RegisterExtension(extension_field) method.
 |        Not sure how to best express at this point.
 |  
 |  DESCRIPTOR = <google.protobuf.descriptor.Descriptor object>
 |  
 |  Image = <class 'tensorflow.core.framework.summary_pb2.Image'>
 |      Abstract base class for protocol messages.
 |      
 |      Protocol message classes are almost always generated by the protocol
 |      compiler.  These generated types subclass Message and implement the methods
 |      shown below.
 |      
 |      TODO(robinson): Link to an HTML document here.
 |      
 |      TODO(robinson): Document that instances of this class will also
 |      have an Extensions attribute with __getitem__ and __setitem__.
 |      Again, not sure how to best convey this.
 |      
 |      TODO(robinson): Document that the class must also have a static
 |        RegisterExtension(extension_field) method.
 |        Not sure how to best express at this point.
 |  
 |  VALUE_FIELD_NUMBER = 1
 |  
 |  Value = <class 'tensorflow.core.framework.summary_pb2.Value'>
 |      Abstract base class for protocol messages.
 |      
 |      Protocol message classes are almost always generated by the protocol
 |      compiler.  These generated types subclass Message and implement the methods
 |      shown below.
 |      
 |      TODO(robinson): Link to an HTML document here.
 |      
 |      TODO(robinson): Document that instances of this class will also
 |      have an Extensions attribute with __getitem__ and __setitem__.
 |      Again, not sure how to best convey this.
 |      
 |      TODO(robinson): Document that the class must also have a static
 |        RegisterExtension(extension_field) method.
 |        Not sure how to best express at this point.
 |  
 |  _decoders_by_tag = {b'\n': (<function MessageDecoder.<locals>.DecodeRe...
 |  
 |  _extensions_by_name = {}
 |  
 |  _extensions_by_number = {}
 |  
 |  ----------------------------------------------------------------------
 |  Methods inherited from google.protobuf.message.Message:
 |  
 |  ClearExtension(self, extension_handle)
 |  
 |  CopyFrom(self, other_msg)
 |      Copies the content of the specified message into the current message.
 |      
 |      The method clears the current message and then merges the specified
 |      message using MergeFrom.
 |      
 |      Args:
 |        other_msg: Message to copy into the current one.
 |  
 |  HasExtension(self, extension_handle)
 |  
 |  ParseFromString(self, serialized)
 |      Parse serialized protocol buffer data into this message.
 |      
 |      Like MergeFromString(), except we clear the object first and
 |      do not return the value that MergeFromString returns.
 |  
 |  __deepcopy__(self, memo=None)
 |  
 |  __getstate__(self)
 |      Support the pickle protocol.
 |  
 |  __hash__(self)
 |      Return hash(self).
 |  
 |  __ne__(self, other_msg)
 |      Return self!=value.
 |  
 |  __setstate__(self, state)
 |      Support the pickle protocol.
- SummaryDescription
Help on class SummaryDescription in module tensorflow.core.framework.summary_pb2:

class SummaryDescription(google.protobuf.message.Message)
 |  Abstract base class for protocol messages.
 |  
 |  Protocol message classes are almost always generated by the protocol
 |  compiler.  These generated types subclass Message and implement the methods
 |  shown below.
 |  
 |  TODO(robinson): Link to an HTML document here.
 |  
 |  TODO(robinson): Document that instances of this class will also
 |  have an Extensions attribute with __getitem__ and __setitem__.
 |  Again, not sure how to best convey this.
 |  
 |  TODO(robinson): Document that the class must also have a static
 |    RegisterExtension(extension_field) method.
 |    Not sure how to best express at this point.
 |  
 |  Method resolution order:
 |      SummaryDescription
 |      google.protobuf.message.Message
 |      builtins.object
 |  
 |  Methods defined here:
 |  
 |  ByteSize(self)
 |  
 |  Clear = _Clear(self)
 |  
 |  ClearField(self, field_name)
 |  
 |  DiscardUnknownFields = _DiscardUnknownFields(self)
 |  
 |  FindInitializationErrors(self)
 |      Finds required fields which are not initialized.
 |      
 |      Returns:
 |        A list of strings.  Each string is a path to an uninitialized field from
 |        the top-level message, e.g. "foo.bar[5].baz".
 |  
 |  HasField(self, field_name)
 |  
 |  IsInitialized(self, errors=None)
 |      Checks if all required fields of a message are set.
 |      
 |      Args:
 |        errors:  A list which, if provided, will be populated with the field
 |                 paths of all missing required fields.
 |      
 |      Returns:
 |        True iff the specified message has all required fields set.
 |  
 |  ListFields(self)
 |  
 |  MergeFrom(self, msg)
 |  
 |  MergeFromString(self, serialized)
 |  
 |  SerializePartialToString(self)
 |  
 |  SerializeToString(self)
 |  
 |  SetInParent = Modified(self)
 |      Sets the _cached_byte_size_dirty bit to true,
 |      and propagates this to our listener iff this was a state change.
 |  
 |  WhichOneof(self, oneof_name)
 |      Returns the name of the currently set field inside a oneof, or None.
 |  
 |  _InternalParse = InternalParse(self, buffer, pos, end)
 |  
 |  _InternalSerialize = InternalSerialize(self, write_bytes)
 |  
 |  _Modified = Modified(self)
 |      Sets the _cached_byte_size_dirty bit to true,
 |      and propagates this to our listener iff this was a state change.
 |  
 |  _SetListener(self, listener)
 |  
 |  _UpdateOneofState(self, field)
 |      Sets field as the active field in its containing oneof.
 |      
 |      Will also delete currently active field in the oneof, if it is different
 |      from the argument. Does not mark the message as modified.
 |  
 |  __eq__(self, other)
 |  
 |  __init__ = init(self, **kwargs)
 |  
 |  __reduce__(self)
 |  
 |  __repr__(self)
 |  
 |  __str__(self)
 |  
 |  __unicode__(self)
 |  
 |  ----------------------------------------------------------------------
 |  Static methods defined here:
 |  
 |  FromString(s)
 |  
 |  RegisterExtension(extension_handle)
 |      # TODO(robinson): This probably needs to be thread-safe(?)
 |  
 |  ----------------------------------------------------------------------
 |  Data descriptors defined here:
 |  
 |  __weakref__
 |      list of weak references to the object (if defined)
 |  
 |  _cached_byte_size
 |  
 |  _cached_byte_size_dirty
 |  
 |  _fields
 |  
 |  _is_present_in_parent
 |  
 |  _listener
 |  
 |  _listener_for_children
 |  
 |  _oneofs
 |  
 |  _unknown_fields
 |  
 |  type_hint
 |      Magic attribute generated for "type_hint" proto field.
 |  
 |  ----------------------------------------------------------------------
 |  Data and other attributes defined here:
 |  
 |  DESCRIPTOR = <google.protobuf.descriptor.Descriptor object>
 |  
 |  TYPE_HINT_FIELD_NUMBER = 1
 |  
 |  _decoders_by_tag = {b'\n': (<function StringDecoder.<locals>.DecodeFie...
 |  
 |  _extensions_by_name = {}
 |  
 |  _extensions_by_number = {}
 |  
 |  ----------------------------------------------------------------------
 |  Methods inherited from google.protobuf.message.Message:
 |  
 |  ClearExtension(self, extension_handle)
 |  
 |  CopyFrom(self, other_msg)
 |      Copies the content of the specified message into the current message.
 |      
 |      The method clears the current message and then merges the specified
 |      message using MergeFrom.
 |      
 |      Args:
 |        other_msg: Message to copy into the current one.
 |  
 |  HasExtension(self, extension_handle)
 |  
 |  ParseFromString(self, serialized)
 |      Parse serialized protocol buffer data into this message.
 |      
 |      Like MergeFromString(), except we clear the object first and
 |      do not return the value that MergeFromString returns.
 |  
 |  __deepcopy__(self, memo=None)
 |  
 |  __getstate__(self)
 |      Support the pickle protocol.
 |  
 |  __hash__(self)
 |      Return hash(self).
 |  
 |  __ne__(self, other_msg)
 |      Return self!=value.
 |  
 |  __setstate__(self, state)
 |      Support the pickle protocol.


- TaggedRunMetadata
Help on class SummaryDescription in module tensorflow.core.framework.summary_pb2:

class SummaryDescription(google.protobuf.message.Message)
 |  Abstract base class for protocol messages.
 |  
 |  Protocol message classes are almost always generated by the protocol
 |  compiler.  These generated types subclass Message and implement the methods
 |  shown below.
 |  
 |  TODO(robinson): Link to an HTML document here.
 |  
 |  TODO(robinson): Document that instances of this class will also
 |  have an Extensions attribute with __getitem__ and __setitem__.
 |  Again, not sure how to best convey this.
 |  
 |  TODO(robinson): Document that the class must also have a static
 |    RegisterExtension(extension_field) method.
 |    Not sure how to best express at this point.
 |  
 |  Method resolution order:
 |      SummaryDescription
 |      google.protobuf.message.Message
 |      builtins.object
 |  
 |  Methods defined here:
 |  
 |  ByteSize(self)
 |  
 |  Clear = _Clear(self)
 |  
 |  ClearField(self, field_name)
 |  
 |  DiscardUnknownFields = _DiscardUnknownFields(self)
 |  
 |  FindInitializationErrors(self)
 |      Finds required fields which are not initialized.
 |      
 |      Returns:
 |        A list of strings.  Each string is a path to an uninitialized field from
 |        the top-level message, e.g. "foo.bar[5].baz".
 |  
 |  HasField(self, field_name)
 |  
 |  IsInitialized(self, errors=None)
 |      Checks if all required fields of a message are set.
 |      
 |      Args:
 |        errors:  A list which, if provided, will be populated with the field
 |                 paths of all missing required fields.
 |      
 |      Returns:
 |        True iff the specified message has all required fields set.
 |  
 |  ListFields(self)
 |  
 |  MergeFrom(self, msg)
 |  
 |  MergeFromString(self, serialized)
 |  
 |  SerializePartialToString(self)
 |  
 |  SerializeToString(self)
 |  
 |  SetInParent = Modified(self)
 |      Sets the _cached_byte_size_dirty bit to true,
 |      and propagates this to our listener iff this was a state change.
 |  
 |  WhichOneof(self, oneof_name)
 |      Returns the name of the currently set field inside a oneof, or None.
 |  
 |  _InternalParse = InternalParse(self, buffer, pos, end)
 |  
 |  _InternalSerialize = InternalSerialize(self, write_bytes)
 |  
 |  _Modified = Modified(self)
 |      Sets the _cached_byte_size_dirty bit to true,
 |      and propagates this to our listener iff this was a state change.
 |  
 |  _SetListener(self, listener)
 |  
 |  _UpdateOneofState(self, field)
 |      Sets field as the active field in its containing oneof.
 |      
 |      Will also delete currently active field in the oneof, if it is different
 |      from the argument. Does not mark the message as modified.
 |  
 |  __eq__(self, other)
 |  
 |  __init__ = init(self, **kwargs)
 |  
 |  __reduce__(self)
 |  
 |  __repr__(self)
 |  
 |  __str__(self)
 |  
 |  __unicode__(self)
 |  
 |  ----------------------------------------------------------------------
 |  Static methods defined here:
 |  
 |  FromString(s)
 |  
 |  RegisterExtension(extension_handle)
 |      # TODO(robinson): This probably needs to be thread-safe(?)
 |  
 |  ----------------------------------------------------------------------
 |  Data descriptors defined here:
 |  
 |  __weakref__
 |      list of weak references to the object (if defined)
 |  
 |  _cached_byte_size
 |  
 |  _cached_byte_size_dirty
 |  
 |  _fields
 |  
 |  _is_present_in_parent
 |  
 |  _listener
 |  
 |  _listener_for_children
 |  
 |  _oneofs
 |  
 |  _unknown_fields
 |  
 |  type_hint
 |      Magic attribute generated for "type_hint" proto field.
 |  
 |  ----------------------------------------------------------------------
 |  Data and other attributes defined here:
 |  
 |  DESCRIPTOR = <google.protobuf.descriptor.Descriptor object>
 |  
 |  TYPE_HINT_FIELD_NUMBER = 1
 |  
 |  _decoders_by_tag = {b'\n': (<function StringDecoder.<locals>.DecodeFie...
 |  
 |  _extensions_by_name = {}
 |  
 |  _extensions_by_number = {}
 |  
 |  ----------------------------------------------------------------------
 |  Methods inherited from google.protobuf.message.Message:
 |  
 |  ClearExtension(self, extension_handle)
 |  
 |  CopyFrom(self, other_msg)
 |      Copies the content of the specified message into the current message.
 |      
 |      The method clears the current message and then merges the specified
 |      message using MergeFrom.
 |      
 |      Args:
 |        other_msg: Message to copy into the current one.
 |  
 |  HasExtension(self, extension_handle)
 |  
 |  ParseFromString(self, serialized)
 |      Parse serialized protocol buffer data into this message.
 |      
 |      Like MergeFromString(), except we clear the object first and
 |      do not return the value that MergeFromString returns.
 |  
 |  __deepcopy__(self, memo=None)
 |  
 |  __getstate__(self)
 |      Support the pickle protocol.
 |  
 |  __hash__(self)
 |      Return hash(self).
 |  
 |  __ne__(self, other_msg)
 |      Return self!=value.
 |  
 |  __setstate__(self, state)
 |      Support the pickle protocol.


help(tf.summary.TaggedRunMetadata)
Help on class TaggedRunMetadata in module tensorflow.core.util.event_pb2:

class TaggedRunMetadata(google.protobuf.message.Message)
 |  Abstract base class for protocol messages.
 |  
 |  Protocol message classes are almost always generated by the protocol
 |  compiler.  These generated types subclass Message and implement the methods
 |  shown below.
 |  
 |  TODO(robinson): Link to an HTML document here.
 |  
 |  TODO(robinson): Document that instances of this class will also
 |  have an Extensions attribute with __getitem__ and __setitem__.
 |  Again, not sure how to best convey this.
 |  
 |  TODO(robinson): Document that the class must also have a static
 |    RegisterExtension(extension_field) method.
 |    Not sure how to best express at this point.
 |  
 |  Method resolution order:
 |      TaggedRunMetadata
 |      google.protobuf.message.Message
 |      builtins.object
 |  
 |  Methods defined here:
 |  
 |  ByteSize(self)
 |  
 |  Clear = _Clear(self)
 |  
 |  ClearField(self, field_name)
 |  
 |  DiscardUnknownFields = _DiscardUnknownFields(self)
 |  
 |  FindInitializationErrors(self)
 |      Finds required fields which are not initialized.
 |      
 |      Returns:
 |        A list of strings.  Each string is a path to an uninitialized field from
 |        the top-level message, e.g. "foo.bar[5].baz".
 |  
 |  HasField(self, field_name)
 |  
 |  IsInitialized(self, errors=None)
 |      Checks if all required fields of a message are set.
 |      
 |      Args:
 |        errors:  A list which, if provided, will be populated with the field
 |                 paths of all missing required fields.
 |      
 |      Returns:
 |        True iff the specified message has all required fields set.
 |  
 |  ListFields(self)
 |  
 |  MergeFrom(self, msg)
 |  
 |  MergeFromString(self, serialized)
 |  
 |  SerializePartialToString(self)
 |  
 |  SerializeToString(self)
 |  
 |  SetInParent = Modified(self)
 |      Sets the _cached_byte_size_dirty bit to true,
 |      and propagates this to our listener iff this was a state change.
 |  
 |  WhichOneof(self, oneof_name)
 |      Returns the name of the currently set field inside a oneof, or None.
 |  
 |  _InternalParse = InternalParse(self, buffer, pos, end)
 |  
 |  _InternalSerialize = InternalSerialize(self, write_bytes)
 |  
 |  _Modified = Modified(self)
 |      Sets the _cached_byte_size_dirty bit to true,
 |      and propagates this to our listener iff this was a state change.
 |  
 |  _SetListener(self, listener)
 |  
 |  _UpdateOneofState(self, field)
 |      Sets field as the active field in its containing oneof.
 |      
 |      Will also delete currently active field in the oneof, if it is different
 |      from the argument. Does not mark the message as modified.
 |  
 |  __eq__(self, other)
 |  
 |  __init__ = init(self, **kwargs)
 |  
 |  __reduce__(self)
 |  
 |  __repr__(self)
 |  
 |  __str__(self)
 |  
 |  __unicode__(self)
 |  
 |  ----------------------------------------------------------------------
 |  Static methods defined here:
 |  
 |  FromString(s)
 |  
 |  RegisterExtension(extension_handle)
 |      # TODO(robinson): This probably needs to be thread-safe(?)
 |  
 |  ----------------------------------------------------------------------
 |  Data descriptors defined here:
 |  
 |  __weakref__
 |      list of weak references to the object (if defined)
 |  
 |  _cached_byte_size
 |  
 |  _cached_byte_size_dirty
 |  
 |  _fields
 |  
 |  _is_present_in_parent
 |  
 |  _listener
 |  
 |  _listener_for_children
 |  
 |  _oneofs
 |  
 |  _unknown_fields
 |  
 |  run_metadata
 |      Magic attribute generated for "run_metadata" proto field.
 |  
 |  tag
 |      Magic attribute generated for "tag" proto field.
 |  
 |  ----------------------------------------------------------------------
 |  Data and other attributes defined here:
 |  
 |  DESCRIPTOR = <google.protobuf.descriptor.Descriptor object>
 |  
 |  RUN_METADATA_FIELD_NUMBER = 2
 |  
 |  TAG_FIELD_NUMBER = 1
 |  
 |  _decoders_by_tag = {b'\n': (<function StringDecoder.<locals>.DecodeFie...
 |  
 |  _extensions_by_name = {}
 |  
 |  _extensions_by_number = {}
 |  
 |  ----------------------------------------------------------------------
 |  Methods inherited from google.protobuf.message.Message:
 |  
 |  ClearExtension(self, extension_handle)
 |  
 |  CopyFrom(self, other_msg)
 |      Copies the content of the specified message into the current message.
 |      
 |      The method clears the current message and then merges the specified
 |      message using MergeFrom.
 |      
 |      Args:
 |        other_msg: Message to copy into the current one.
 |  
 |  HasExtension(self, extension_handle)
 |  
 |  ParseFromString(self, serialized)
 |      Parse serialized protocol buffer data into this message.
 |      
 |      Like MergeFromString(), except we clear the object first and
 |      do not return the value that MergeFromString returns.
 |  
 |  __deepcopy__(self, memo=None)
 |  
 |  __getstate__(self)
 |      Support the pickle protocol.
 |  
 |  __hash__(self)
 |      Return hash(self).
 |  
 |  __ne__(self, other_msg)
 |      Return self!=value.
 |  
 |  __setstate__(self, state)
 |      Support the pickle protocol.
- tensor_summary
Help on function tensor_summary in module tensorflow.python.ops.summary_ops:

tensor_summary(name, tensor, summary_description=None, collections=None)
    Outputs a `Summary` protocol buffer with a serialized tensor.proto.
    
    The generated
    [`Summary`](https://www.tensorflow.org/code/tensorflow/core/framework/summary.proto)
    has one summary value containing the input tensor.
    
    Args:
      name: A name for the generated node. Will also serve as the series name in
        TensorBoard.
      tensor: A tensor of any type and shape to serialize.
      summary_description: Optional summary_pb2.SummaryDescription()
      collections: Optional list of graph collections keys. The new summary op is
        added to these collections. Defaults to `[GraphKeys.SUMMARIES]`.
    
    Returns:
      A scalar `Tensor` of type `string`. The serialized `Summary` protocol
      buffer.

