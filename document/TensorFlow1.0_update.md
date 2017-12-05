# TensorFLow1.0 更新函数
## Variables
tf.VARIABLES----tf.GLOBAL_VARIABLES  
tf.all_variables ---- tf.GLOBAL_VARIABLES  
tf.initialize_all_variables ---- tf.global_variables_initializer  
tf.inilize_local-variables ---- tf.local_variables_initializer  
tf.initialize_variables ---- tf.variables_initializer  

## Summary functions
tf.audio_summary ---- tf.Summary.audio  
tf.contrib.deprecated.histgram_summary ---- tf.Summary.histgram  
tf.image_summary ---- tf.summary.image  
tf.merge_summary ---- tf.summary.merge  
tf.scalar_summary ---- tf.summary.scalar  
tf.train.SummaryWriter ---- tf.summary.FileWriter  

# Numeric differences
|Expr|TF0.11(Py2)| 0.11(Py3)|TF 1.0(py2)|TF 1.0(py3)|
|:-----:|:-----:|:--------:|:--------:|:----------:|
|tf.div(3,4)|0|0|0|0|
|tf.div(-3,4)|0|0|-1|-1|
|td.mod(-3,4)|-3|-3|1|1|
|-3/4|0|-0.75|-1|-0.75|
|-3/4 tf.divide(-3,4)|N/A|N/A|-0.75|-1|  
Round函数  

|Input|Python|C++ round()|Tensorflow 0.11(floor(x+.5))|Tensorflow 1.0|
|:----:|:----:|:----:|:----:|:----:|:------:|
|-3.5|-4|-4|-4|-3|-4|
|-2.5|-2|-2|-3|-2|-2|
|-1.5|-2|-2|-2|-1|-2|
|-0.5|0|0|-1|0|0|
|0.5|0|0|1|1|0|
|1.5|2|2|2|2|2|
|2.5|2|2|3|3|2|
|3.5|4|4|4|4|4|

# Numpy matching names
- tf.inv
  - should be rename to tf.reciprocal
  - 避免和Numpy的矩阵求逆np.inv混淆
- tf.list_diff
  - 被命名为tf.setdiffld
- tf.list_diff
  - 被命名为tf.setdiff1l
- tf.mul
  - 命名为tf.multiply
- tf.neg
  - 被命名为tf.negative
- tf.select
  - 被命名为tf.where
  - tf.where  now take 3 arguments or 1 arguments,just line np.where
- tf.sub
  - 命名为tf.substruct
# Numpy matching arhuments
- tf.argmax
  - 关键字dimension被命名为axis
- tf.argmin
  - 关键字dimension被命名为axis
- tf.concat
  - 关键字concat_dim被命名为axis
  - 参数顺序将被设置为tf.concat(values,axis,name='concat')
- tf.count_nonzero
  - 关键字reduction_indices被命名为axis
- tf.expand_dims
  - 关键字dim被命名为axis
- tf.reduce_all
  - 关键字reduction_indices将被命名为axis
- tf.reduce_any
  - 关键字reduction_indices将被命名为axis
- tf.reduce_join
  - 关键字reduction_indices将被命名为axis
- tf.reduce_logsumexp
  - 关键字reduction_indices将被命名为axis
- tf.reduce_max
  - 关键字reduction_indices将被命名为axis
- tf.reduce_mean
  - 关键字reduction_indices将被命名为axis
- tf.reduce_min
  - 关键字reduction_indices将被命名为axis
- tf.reduce_prod
  - 关键字reduction_indices将被命名为axis
- tf.reverse
  - tf.reverse 过去用以为数字向量控制reversed的维度，现在我们用axis作为索引。
  - 例如 tf.reverse(a,[True,False,True])现在写作tf.reverse(a,[0,2])
- tf.reverse_sequence
  - 关键字batch_dim 应该被命名为batch_axis
  - 关键字seq_dim应该被命名为seq_axis
- tf.sparse_concat
  - 关键字concat_dim应该被命名为axis
- tf.sparse_reduce_sum
  - 关键字reduction_axis应该被命名为axis
- tf.sparse_reduce_sum_sparse
  - 关键字reduction_axis应该被命名为axis
- tf.sparse_split
  - 关键字split_dim应该被命名为axis
  - 新的参数顺序tf.sparse_split(keyword_required=KeywordRequired(),sp_input=None,num_split = None,axis=None,name=None,split_dim=None)
  - tf.sparse_split
    - 关键字split_dim将被命名为axis
    - 关键字num_split将被命名为num_or_size_splits
    - 参数顺序tf.split(value,nun_or_size_splits,axis=0,num=None,name='split')
- tf.squeeze
  - 关键字squeeze_dims应该被命名为axis
- tf.svd
  - 参数顺序tf.svd(tensor,full_matrices=False,computer_uv=True,name=None)
  # Simplified math variants
  批量版本的数学操作将被移除。
  - tf.batch_band_part
    - 应给命名为tf.band_part
- tf.batch_cholesky
  - 应该被命名为tf.cholesky
- tf.batch_fft
  - 应该被命名为tf.fft
- tf.batch_fft3d
  - 应该被命名为tf.fft3d
- tf.batch_ifft
  -  应该被命名为tf.fft
- tf.batch_ifft2d
  - 应该被命名为tf.ifft2d
- tf.batch_ifft3d
  - 应该被命名为 tf.ifft3d
- tf.batch_matmul
  - 应该被命名为tf.matmul
- tf.batch_matrix_determinant
  - 应该被命名为tf.matrix_determinant
- tf.batch_matrix_inverse
  - 应该被命名为tf.matrix_inverse
- tf.batch_matrix_solve
  - 应该被命名为tf.matrix_solve
- tf.batch_matrix_solve_ls
  - 应该被命名为tf.matrix_solve_ls
- tf.batch_matrix_transpose
  - 应该被命名为tf.matrix_transpose
- tf.batch_matrix_triangular_solve
  - 应该被命名为tf.matrix_triangeular_solve
- tf.bath_self_adjoint_eig
  - 应该被命名为tf.self.adjoint_eig
- tf.batch_self_adjoint_eigvals
  - 应该被命名为 tf.self_adjoint_eigvals
- tf.batch_set_diag
  - 应该被命名为tf.set_diag
- tf.batch_svd
  - 应该被命名为tf.svd
- tf.complex_abs
 - 应该被命名为tf.abs
# Misc Changes
- tf.image.per_image_whitening
  - 应该被命名为tf.image.per_image_standardization
- tf.nn.sigmoid_cross_entropy_with_logits
  - 参数顺序变为tf.nn.sigmoid_cross_entropy_with_logits(_setinel=None,label=Non,logits = None,dim = -1,name=None)
- tf.nn.softmax_cross_entropy_with_logits
  - 参数顺序为tf.nn.softmax_cross_entropy_with_logits(_setinel=None,label=Non,logits = None,dim = -1,name=None)
  - tf.nn.sparse_softmax_cross_entropy_with_logits
    - 参数顺序为tf.nn.sparse_softmax_cross_entropy_with_logits(_setinel=None,label=Non,logits = None,dim = -1,name=None)
  - tf.one_initializer
    - 应该调用tf.one_initilizer()
  - tf.pack
    - 应该命名为tf.stack
  - tf.Round
    - tf.round现在匹配Banker's rounding
  - tf.upstack
    - 应该被命名为tf.unstack
  - tf.zeros_initializer
    - 应该调用tf.zeros_initializer()
## Tensorflow Board运行
进入包含tensorboard的文件夹的上一级文件夹，在powershell下输入tensorboard --logdir 'log/'
