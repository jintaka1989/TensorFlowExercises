# -*- coding: utf-8 -*-
# 未完成
# ニューラルネットワークが任意の関数を表現できることの視覚的証明
# http://nnadl-ja.github.io/nnadl_site_ja/chap4.html
# 上記を実装したい

import tensorflow as tf
import numpy as np
import time
import matplotlib.pyplot as plt

start = time.time()

# data set
x_sample = np.random.rand(100,1).astype("float32")
y_sample = 0.1 * x_sample * x_sample + 0.3
# y_sample = y_sample + 0.01*np.random.rand(100,1).astype("float32")

x_data = tf.placeholder(tf.float32,[1])
y_data = tf.placeholder(tf.float32,[1])


# 素子の数（テンソルの数）を決める
tensor_number = (2,100)
range_number = tensor_number[0]*tensor_number[1]
w_define = 1000

# # # -b_input/w_input = s(s=0.001*n)
# w1 = tf.constant(range(100),shape=tensor_number[0:], dtype=tf.float32) * 0.01
# w2 = tf.constant(range(100),shape=tensor_number[1:], dtype=tf.float32) * (-0.01)
w = tf.Variable(tf.constant(w_define, shape=tensor_number, dtype=tf.float32))
b = tf.Variable(tf.constant(range(range_number),shape=tensor_number, dtype=tf.float32) * 10 -1000)

w_output = tf.Variable(tf.random_uniform(tensor_number, -1.0, 1.0))
# b_output = tf.Variable(tf.zeros(tensor_number))

# # hidden layer
# h_linear1 = tf.add(w_input*x_data, b_input)
h_linear1 = tf.add(x_data*w, b)

# # output layer
h_linear3 = tf.reduce_sum(tf.matmul(h_linear1,w_output, transpose_a = True))
# h_linear3 = tf.reduce_sum(tf.batch_matmul(h_linear2,w_output, adj_x=True, adj_y=True))

y = h_linear3

loss = tf.reduce_mean(tf.square(y_data - y))

# Outputs a Summary protocol buffer with scalar values
loss_summary = tf.scalar_summary("loss", loss)

optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)

init = tf.initialize_all_variables()

sess = tf.Session()
merged = tf.merge_all_summaries()
writer = tf.train.SummaryWriter("/tmp/tensorflow_log", sess.graph_def)
sess.run(init)

for step in xrange(1001):
    for i in xrange(100):
        if step % 100 == 0:
            result = sess.run([merged, loss],feed_dict={x_data:x_sample[i], y_data:y_sample[i]})
            summary_str = result[0]
            acc = result[1]
            writer.add_summary(summary_str, step)
            print step
            print sess.run(w_output)
            print step
        else:
            sess.run(train, feed_dict={x_data:x_sample[i], y_data:y_sample[i]})

xx = np.arange(0, 1, 0.1)
yy = 0.1 * xx * xx + 0.3

plt.plot(xx, yy)

x=[]
result = []

for i in xrange(20):
    prot_x = (i)*0.05
    x.append(prot_x)
    result.append(sess.run(y, feed_dict={x_data:[prot_x]}))
    print result[i]

plt.scatter(x, result)

plt.show()

sess.close()

timer = time.time() - start
print ("time:{0}".format(timer)) + "[sec]"
