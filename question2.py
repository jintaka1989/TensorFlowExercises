# -*- coding: utf-8 -*-
# 一次関数の近似を行いたい
# y_sample = 0.1*x_sample + 0.3
# 演習のためテンソルを増やしてみる
# 入力層１個、隠れ層4*4個、出力層１個のモデルを作成せよ
# それぞれ重みWとバイアスbを使うこと

import tensorflow as tf
import numpy as np
import time
import matplotlib.pyplot as plt

start = time.time()

# data set
x_sample = np.random.rand(100,1).astype("float32")
y_sample = 0.1 * x_sample + 0.3
y_sample = y_sample + 0.01*np.random.rand(100,1).astype("float32")

x_data = tf.placeholder(tf.float32,[1])
y_data = tf.placeholder(tf.float32,[1])

w_input = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
b_input = tf.Variable(tf.zeros([1]))

# 素子の数（テンソルの数）を決める
tensor_number = [4,4]
w = tf.Variable(tf.random_uniform(tensor_number, -1.0, 1.0))
b = tf.Variable(tf.zeros(tensor_number))

w_output = tf.Variable(tf.random_uniform(tensor_number, -1.0, 1.0))
b_output = tf.Variable(tf.zeros([1]))

# input layer
h_linear1 = tf.add(w_input*x_data, b_input)

# first hidden layer
h_linear2 = tf.add(h_linear1*w, b)
# h_linear2 = h_linear1*w + b
# # output layer
h_linear3 = tf.reduce_sum(tf.matmul(h_linear2,w_output, transpose_a = True)) + b_output
# h_linear3 = tf.reduce_sum(tf.batch_matmul(h_linear2,w_output, adj_x=True, adj_y=True)) + b_output

y = h_linear3

loss = tf.reduce_mean(tf.square(y_data - y))

# Outputs a Summary protocol buffer with scalar values
loss_summary = tf.scalar_summary("loss", loss)

optimizer = tf.train.GradientDescentOptimizer(0.02)
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
            print sess.run(w)
            print sess.run(b)
        else:
            sess.run(train, feed_dict={x_data:x_sample[i], y_data:y_sample[i]})

xx = np.arange(-10, 10, 1)
yy = 0.1 * xx + 0.3

plt.plot(xx, yy)

x=[]
result = []

for i in xrange(20):
    prot_x = i-10
    x.append(prot_x)
    result.append(sess.run(y, feed_dict={x_data:[prot_x]}))
    print result[i]

plt.scatter(x, result)

plt.show()

sess.close()

timer = time.time() - start
print ("time:{0}".format(timer)) + "[sec]"
