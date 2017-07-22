import tensorflow as tf
import matplotlib.pyplot as plt
x = tf.linspace(-10.,10.,100)
y = tf.nn.relu(x)
with tf.Session() as sess:
    [x,y] = sess.run([x,y])

plt.plot(x,y,'r')
plt.xlabel('x')
plt.ylabel('relu')
plt.title('relu')
ax = plt.gca()
ax.annotate("",xy=(6, 6), xycoords='data ',
xytext=(6, 4.5), textcoords='data ',
arrowprops=dict(arrowstyle="->",
connectionstyle="arc3"),
)
plt.savefig('relu.png',dpi=600)
