# -*- coding: utf-8 -*-
# sinをn次関数で近似する
# 9次関数を選択した
# データセットが0から1のため,-1~0で誤差が生じる
# データセットを-1まで増やしたものはquestion9_2.py

import tensorflow as tf
import numpy as np
import time
import matplotlib.pyplot as plt
import math

start = time.time()

# number of datum pare in dataset
AC=100
# number of the function s dimention
NFD = 9
#1+number of the function s dimention
Nadd1=NFD+1
# number of W
WN=Nadd1

# 傾きslopeと切片interceptをここに書いて統一
slope = []
for i in xrange(NFD):
    slope.append((i+1)*0.1)
intercept = 1.0

# data set
def create_dataset():
    x_result = []
    y_result = []
    for n in xrange(AC):
        x_sample = np.random.rand(1).astype("float32")
        x_power = []
        for i in xrange(NFD):
            x_power.append(pow(x_sample,(i+1)))
        y_sample = math.sin(x_sample*2)*0.5
        y_sample = y_sample + 0.01*np.random.rand(1).astype("float32")
        x_result.append(x_sample)
        y_result.append(y_sample)
    return x_result, y_result

x_sample, y_sample = create_dataset()

x_data = tf.placeholder(tf.float32,[1])
y_data = tf.placeholder(tf.float32,[1])

w_input = tf.Variable(tf.random_uniform([NFD,1], -1.0, 1.0))
b_input = tf.Variable(tf.zeros([1]))

# 素子の数（テンソルの中身の数）を決める
tensor_number = (1,1,1)

w = tf.Variable(tf.random_uniform(tensor_number, -1.0, 1.0))
b = tf.Variable(tf.zeros(tensor_number))

w_output = tf.Variable(tf.random_uniform(tensor_number, -1.0, 1.0))
b_output = tf.Variable(tf.zeros([1]))

# 内積, 一般式はslope[n]*x^n
def matmul_extend(w_input, x):
    x_power = []
    for i in xrange(NFD):
        x_power.append(pow(x,(i+1)))
    y_sample = tf.matmul(w_input,x_power,transpose_a=True)
    return y_sample

# input layer
h_linear1 = tf.add(matmul_extend(w_input, x_data), b_input)

# first hidden layer
h_linear2 = tf.add(h_linear1*w, b)
# h_linear2 = h_linear1*w + b

# # output layer
# h_linear3 = tf.reduce_sum(tf.matmul(h_linear2,w_output, transpose_a = True)) + b_output
h_linear3 = tf.reduce_sum(tf.batch_matmul(h_linear2,w_output, adj_x=True, adj_y=True)) + b_output

y = h_linear3

loss = tf.reduce_mean(tf.square(y_data - y))

# Outputs a Summary protocol buffer with scalar values
loss_summary = tf.scalar_summary("loss", loss)

optimizer = tf.train.GradientDescentOptimizer(0.1)
train = optimizer.minimize(loss)

init = tf.initialize_all_variables()

sess = tf.Session()
merged = tf.merge_all_summaries()
writer = tf.train.SummaryWriter("/tmp/tensorflow_log", sess.graph_def)
sess.run(init)

for step in xrange(2001):
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
        print sess.run(w_input)
        print sess.run(b_input)


# data set
def create_plotset():
    x_result = []
    y_result = []
    for n in xrange(20):
        x_sample = (n-10)*0.1
        # slope = []
        x_power = []
        for i in xrange(NFD):
            # slope.append((i+1)*0.1)
            x_power.append(pow(x_sample,(i+1)))
        # intercept = 1.0
        y_sample = math.sin(x_sample*2)*0.5
        x_result.append(x_sample)
        y_result.append(y_sample)
    return x_result, y_result

xx, yy = create_plotset()

plt.plot(xx,yy)

x=[]
result = []

for i in xrange(20):
    prot_x = (i-10)*0.1
    x.append(prot_x)
    result.append(sess.run(y, feed_dict={x_data:[prot_x]}))
    print result[i]

plt.scatter(x, result)
plt.axhline(0, color='black')
plt.axvline(0, color='black')
plt.show()

sess.close()

timer = time.time() - start
print ("time:{0}".format(timer)) + "[sec]"
