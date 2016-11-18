# -*- coding: utf-8 -*-
# 未完成
# 二次関数の近似を行いたい
# y_sample = 0.1*x_sample*x_sample + 0.3
# 演習のためテンソルを増やしてみる
# 入力層１個、隠れ層(4*4)*2個,出力層１個のモデルを作成せよ
# それぞれ重みWとバイアスbを使うこと

import tensorflow as tf
import numpy as np
import time
import matplotlib.pyplot as plt

start = time.time()

# data set
x_sample = np.random.rand(100,1).astype("float32")
y_sample = 0.1*x_sample*x_sample*x_sample + 0.3
y_sample = y_sample

x_data = tf.placeholder(tf.float32,[1])
y_data = tf.placeholder(tf.float32,[1])

w_input = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
b_input = tf.Variable(tf.zeros([1]))

w = tf.Variable(tf.random_uniform([4,4], -1.0, 1.0))
b = tf.Variable(tf.zeros([4,4]))

w2 = tf.Variable(tf.random_uniform([4,4], -1.0, 1.0), "w2")
b2 = tf.Variable(tf.zeros([4,4]), "b2")

w_output = tf.Variable(tf.random_uniform([4,4], -1.0, 1.0))
b_output = tf.Variable(tf.zeros([1]))

# input layer
h_linear1 = tf.add(w_input*x_data*x_data, b_input)

# first hidden layer
h_linear2 = tf.add(h_linear1*w, b)

# first hidden layer
h_linear3 = tf.matmul(h_linear2, w2) + b2

# output layer
h_linear_output = tf.reduce_sum(tf.matmul(h_linear2,w_output)) + b_output

y = h_linear_output

# Add summary ops to collect data
w_hist = tf.histogram_summary("weights2", w2)
b_hist = tf.histogram_summary("biases2", b2)
y_hist = tf.histogram_summary("y", y)

loss = tf.reduce_mean(tf.square(y_data - y))
loss_summary = tf.scalar_summary("loss", loss)

optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)

init = tf.initialize_all_variables()

sess = tf.Session()
merged = tf.merge_all_summaries()
writer = tf.train.SummaryWriter("/tmp/tensorflow_log", sess.graph_def)
sess.run(init)

for step in xrange(10001):
    for i in xrange(100):
        if step % 100 == 0:
            result = sess.run([merged, loss],feed_dict={x_data:x_sample[i], y_data:y_sample[i]})
            summary_str = result[0]
            acc = result[1]
            writer.add_summary(summary_str, step)
        else:
            sess.run(train, feed_dict={x_data:x_sample[i], y_data:y_sample[i]})
    if step % 100 == 0:
        print step
        print sess.run(w2)
        print sess.run(b2)

xx = np.arange(-10, 10, 1)
yy = 0.1 * xx * xx* xx + 0.3

plt.plot(xx, yy)

x=[]
r = []

for i in xrange(20):
    prot_x = i-10
    x.append(prot_x)
    r.append(sess.run(y, feed_dict={x_data:[prot_x]}))
    print(r[i])

plt.scatter(x, r)

plt.show()

sess.close()

timer = time.time() - start
print(("time:{0}".format(timer)) + "[sec]")
