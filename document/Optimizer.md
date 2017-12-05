# 这个文件用来介绍关于优化器和他们的性能分析
## Batch gradient descent
$$\theta = \theta-\eta\cdot\Delta_{\theta}J(\theta)$$
通过学习速率界定沿着梯度方向更新的大小，如果是凸函数，批量梯度算法收敛到全局最小值，如果是非凸曲面，收敛到局部最小值。
在代码中批量梯度算法按照如下更新
```Python
for i in range(bn_epochs):
  params_grad = evaluate_gradient(loss_function,data,params)
  params = params -learning_rate*params_grad

```
# Stochastic gradent descent
$$\theta = \theta-\eta\cdot\Delta_{\theta}J(\theta;x^{(i)};y^{(i)})$$
批量梯度算法对大数据更新的时候执行了多余的计算它在更新每个参数的时候计算了类似样本的梯度。SGD通过计算一次同时更新来执行，通常SGD更快，SGD执行时快速更新，伴随着搞得Variance，导致函数波动剧烈。
当参数被替换的时候批量梯度算法收敛到参数的最小值，SGD的波动在另一方面，肯能跳到新的可能是更好的局部最小值，换种说法，最终收敛到确切的最小值。当我满满的减小学习速率的时候，SGD显示了和批量梯度算法相似的收敛性，，如果在凸平面上收敛到全局最小值，非凸平面收敛到局部最小值。
```Python
for i in range(nb_epochs):
  np.random.shuffle(data)
  for example in data:
    params_grad = evaluate_gradient(loss_function,example,params)
    params = params - learn_rate(params_grad)

```
# Mini-batch gradient descent
Mini批量梯度算法一次更新n个训练样本。
$$\theta = \theta=\eta\cdots\Delta_{\theta}J(\theta;x^{(i:i+n)};y^{(i:i+n)})$$
通过这种方法：
1. 减少参数更新的variance，能更稳定的收敛。
2. 可以利用深度学习库高度优化的矩阵优化器计算梯度，通常mini-batch大小在50-256。
mini-batch可以有多种用途，时神经网训练常采用的的方法同时SGD通常也结合一起采用，下面是修改的SGD，我们给出参数$x^{i:i+n};y^{i:i+n}$
z在下面的代码中我们通过mini-batch(50)大小迭代
```Python
for i in range(nb_epochs):
    np.random.shuffle(data)
    for batch in get_batch(data,batch_size=50):
      params_grad = evaluate_gradient(loss_function,batch_size,params)
      params = params -learning_rate * params_grad
```
