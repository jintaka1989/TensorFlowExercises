# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
import time
import random
import math
import matplotlib.pyplot as plt

# import pdb; pdb.set_trace()

# # data set
# x_sample = (np.random.rand(100,1)*10 - 5).astype("float32")
# y_sample  = 0.4*pow(x_sample, 4) + 0.3*pow(x_sample, 3) + 0.2*pow(x_sample, 2) + 0.1*x_sample + 1
#
# plt.scatter(x_sample, y_sample)
#
# plt.show()

# # data set
# x_sample = (np.random.rand(100,1)*10 - 5).astype("float32")
# y_sample  = 0.3*pow(x_sample, 3) + 0.2*pow(x_sample, 2) + 0.1*x_sample + 1
#
# plt.scatter(x_sample, y_sample)
#
# plt.show()

start = time.time()
AC=100
# number of W
WN=3

#1+number of the function s dimention
NN=3
a = np.arange(0.1,(WN+1)*0.1,0.1)

def y_from_x(_x,_W,_b):
    _y = np.dot(_W,_x) + _b
    return _y

x_data = np.random.rand(AC,NN,1).astype("float32")
y_data = np.zeros((AC,NN,1)).astype("float32")
npow = 1

for i in xrange(NN):
    y_data += a[i] * npow
    npow *= x_data

W = tf.Variable(tf.random_uniform([WN], -1.0, 1.0))
y = 0
npow = 1


for i in xrange(WN):
    y += W[i] * npow
    npow *= x_data

loss = tf.reduce_mean(tf.square(y_data - y))
optimizer = tf.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(loss)

init = tf.initialize_all_variables()

sess = tf.Session()
sess.run(init)

for step in xrange(5001):
    sess.run(train)
    if step % 100 == 0:
        print "training...", sess.run(W)

plot_x = []
plot_y = []

for i in xrange(100):
    result = 0
    npow = 1
    for j in xrange(WN):
        result += W[j] * npow
        npow *= i
    plot_x.append(i)
    plot_y.append(sess.run(result))
    print i, sess.run(result)

x_correct = np.arange(-100, 100, 1)
y_correct =0.3*pow(x_correct, 3) + 0.2*pow(x_correct, 2) + 0.1*x_correct + 1

plt.plot(x_correct, y_correct)
plt.scatter(plot_x, plot_y)

plt.show()

sess.close()

timer = time.time() - start

print ("time:{0}".format(timer)) + "[sec]"
