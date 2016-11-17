# -*- coding: utf-8 -*-
# 一次関数の近似を行いたい
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

print x_sample

# x_data = np.random.rand()
x_data = tf.placeholder(tf.float32,[1])
# y_data = 0.1 * x_data + 0.3
# # y_sample = y_data + 0.01*np.random.rand()
# y_data = y_data + 0.01*np.random.rand()
y_data = tf.placeholder(tf.float32,[1])

w_input = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
b_input = tf.Variable(tf.zeros([1]))


w = tf.Variable(tf.random_uniform([4,4], -1.0, 1.0))
b = tf.Variable(tf.zeros([4,4]))

w_output = tf.Variable(tf.random_uniform([4,4], -1.0, 1.0))
b_output = tf.Variable(tf.zeros([1]))

# input layer
h_linear1 = tf.add(w_input*x_data, b_input)

print h_linear1

# first hidden layer
h_linear2 = tf.add(h_linear1*w, b)
# h_linear2 = h_linear1*w + b

# output layer
h_linear3 = tf.reduce_sum(tf.matmul(h_linear2,w_output)) + b_output

y = h_linear3

loss = tf.reduce_mean(tf.square(y_data - y))
optimizer = tf.train.GradientDescentOptimizer(0.02)
train = optimizer.minimize(loss)

# クロスエントロピーはクラス分類に向いている
# cross_entropy = -tf.reduce_sum(y_data*tf.log(y))
# train = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)


init = tf.initialize_all_variables()

sess = tf.Session()
sess.run(init)

for step in xrange(2001):
    for i in xrange(100):
        sess.run(train, feed_dict={x_data:x_sample[i], y_data:y_sample[i]})

    if step % 100 == 0:
        print step
        print sess.run(w)
        print sess.run(b)

xx = np.arange(-100, 100, 1)
yy = 0.1 * xx + 0.3

plt.plot(xx, yy)

# x = np.ndarray(shape=(1,100))

x=[]
result = []

for i in xrange(200):
    x.append(i)
    result.append(sess.run(y, feed_dict={x_data:[i]}))
    print result[i]

plt.scatter(x, result)

plt.show()

sess.close()

timer = time.time() - start
print ("time:{0}".format(timer)) + "[sec]"
