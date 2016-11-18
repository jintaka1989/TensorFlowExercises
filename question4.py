# -*- coding: utf-8 -*-
# 未完成
# ３次関数の近似を行いたい
# y_sample = 0.4*pow(x_sample, 4) + 0.3*pow(x_sample, 3) + 0.2*pow(x_sample, 2) + 0.1

import tensorflow as tf
import numpy as np
import time
import matplotlib.pyplot as plt

start = time.time()

# data set
x_sample = (np.random.rand(100,1)*10 - 5).astype("float32")
y_sample  = 0.3*pow(x_sample, 3) + 0.2*pow(x_sample, 2) + 0.1*x_sample + 1
# y_sample = y_sample

# import pdb; pdb.set_trace()

x_data = tf.placeholder(tf.float32,[1])
y_data = tf.placeholder(tf.float32,[1])

# w_input = tf.Variable(tf.random_uniform([1,3], -1.0, 1.0))
w_input = tf.Variable(tf.ones([1,3]))
b_input = tf.Variable(tf.ones([1]))

pow_x_n_list = []

# one layer
for i in xrange(3):
    pow_x_n_list.append(pow(x_data, i))

h_linear1 = tf.matmul(w_input , pow_x_n_list, transpose_a=False) + b_input

y = h_linear1

loss = tf.reduce_mean(tf.square(y_data - y))
# loss = -tf.reduce_sum(y_data*tf.log(y))

optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)

init = tf.initialize_all_variables()

sess = tf.Session()
# # Add summary ops to collect data
# w_hist = tf.histogram_summary("w_input", w_input)
# b_hist = tf.histogram_summary("b_input", b_input)
# y_hist = tf.histogram_summary("y", y)
# loss_summary = tf.scalar_summary("loss", loss)
# merged = tf.merge_all_summaries()
# writer = tf.train.SummaryWriter("/tmp/tensorflow_log", sess.graph_def)
sess.run(init)

for step in xrange(1001):
    for i in xrange(100):
        if step % 100 == 0:
            # result = sess.run([merged, loss],feed_dict={x_data:x_sample[i], y_data:y_sample[i]})
            # summary_str = result[0]
            # acc = result[1]
            # writer.add_summary(summary_str, step)
            sess.run(train, feed_dict={x_data:x_sample[i], y_data:y_sample[i]})
        else:
            sess.run(train, feed_dict={x_data:x_sample[i], y_data:y_sample[i]})
    if step % 100 == 0:
        print step
        print sess.run(w_input)
        print sess.run(b_input)

x_correct = np.arange(-10, 10, 1)

y_correct = 0.3*pow(x_correct, 3) + 0.2*pow(x_correct, 2) + 0.1

plt.plot(x_correct, y_correct)

x=[]
re = []

for i in xrange(20):
    plot_x = i-10
    x.append(plot_x)
    re.append(sess.run(y, feed_dict={x_data:[plot_x]}))
    print i
    print(re[i])

plt.scatter(x, re)

plt.show()

sess.close()

timer = time.time() - start
print(("time:{0}".format(timer)) + "[sec]")
