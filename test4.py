#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

train_X = np.linspace(-1, 1, 100)
train_Y = 2 * train_X + np.random.randn(*train_X.shape) * 0.33 + 10


print(train_X)
print(train_Y)

with tf.Graph().as_default() as g:
    X = tf.placeholder("float")
    Y = tf.placeholder("float")
    w = tf.Variable(0.0, name="weight")
    b = tf.Variable(0.0, name="bias")
    loss = tf.square(Y - X*w - b)
    train_op = tf.train.GradientDescentOptimizer(0.01).minimize(loss)

with tf.Session(graph=g) as sess:
    sess.run(tf.initialize_all_variables())
    epoch=1
    for i in range(10):
        for(x,y) in zip(train_X,train_Y):
            _,w_value,b_value=sess.run([train_op, w, b],feed_dict={X: x,Y: y})
            print("Epoch: {}, w: {}, b: {}".format(epoch, w_value, b_value))
            epoch += 1
   
#draw
plt.plot(train_X,train_Y,"+")
plt.plot(train_X,train_X.dot(w_value)+b_value)
plt.show()
