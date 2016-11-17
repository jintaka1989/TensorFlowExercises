# -*- coding: utf-8 -*-
# 一次関数の近似を行いたい
# 入力層１個、隠れ層２個、出力層１個のモデルを作成せよ
# それぞれ重みWとバイアスbを使うこと

import tensorflow as tf
import numpy as np
import time
import matplotlib.pyplot as plt

start = time.time()

# data set
x_data = np.random.rand(100).astype("float32")
y_data = 0.1 * x_data + 0.3
y_data = y_data + 0.01*np.random.rand(100).astype("float32")

w_input = tf.Variable(tf.random_uniform([1], -1.0, 1.0), name="w_input")
b_input = tf.Variable(tf.zeros([1]), name="b_input")

W = tf.Variable(tf.random_uniform([2], -1.0, 1.0), name="W_hidden")
b = tf.Variable(tf.zeros([2]), name="b_hidden")

w_output = tf.Variable(tf.random_uniform([2], -1.0, 1.0), name="w_output")
b_output = tf.Variable(tf.zeros([1]), name="b_output")

y_ = w_input * x_data + b_input

y_1 = W[0]*y_ + b[0]
y_2 = W[1]*y_ + b[1]

y = w_output[0]*y_1 + w_output[1]*y_2 + b_output

# Add summary ops to collect data
w_hist = tf.histogram_summary("weights", W)
b_hist = tf.histogram_summary("biases", b)
y_hist = tf.histogram_summary("y", y)

loss = tf.reduce_mean(tf.square(y_data - y))

# Outputs a Summary protocol buffer with scalar values
loss_summary = tf.scalar_summary("loss", loss)

optimizer = tf.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(loss)

# クロスエントロピーはクラス分類に向いている
# cross_entropy = -tf.reduce_sum(y_data*tf.log(y))
# train = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)


init = tf.initialize_all_variables()

sess = tf.Session()

merged = tf.merge_all_summaries()
writer = tf.train.SummaryWriter("/tmp/tensorflow_log", sess.graph_def)

sess.run(init)

w=[]
bias=[]

for step in xrange(1001):
    sess.run(train)
    if step % 10 == 0:
        result = sess.run([merged, loss])
        summary_str = result[0]
        acc = result[1]
        writer.add_summary(summary_str, step)
        print step, sess.run(W), sess.run(b)

# W = sess.run(W)
# b = sess.run(b)
#
# xx = np.arange(-10, 10, 1)
# yy = 0.1 * xx + 0.3
#
# plt.plot(xx, yy)
#
# x = np.arange(-10, 10, 1)
#
# y_ = W[0] * x + b[0]
# y_1 = W[1]*y_ + b[1]
# y_2 = W[2]*y_ + b[2]
# y = W[3]*y_1 + W[4]*y_2 + b[3]
#
# plt.scatter(x, y)
#
# plt.show()

sess.close()

timer = time.time() - start
print ("time:{0}".format(timer)) + "[sec]"
