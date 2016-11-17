# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
import time
import matplotlib.pyplot as plt

x_data = np.random.rand(100).astype("float32")

h_linear1 = w_input*x_data + b_input

h_linear1_tensor = []

for i in xrange(4):
    h_linear1_tensor.append(h_linear1)

h_linear1 = []

for i in xrange(4):
    h_linear1.append(h_linear1_tensor)
