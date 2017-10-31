#!/usr/bin/env python

import tensorflow as tf
import numpy as np


# tf stuff below
with tf.name_scope("name") as scope:

    beta = tf.Variable([0.0, 0.0, 0.0])

    learning_rate = tf.Variable(0.005)

    x_hat = tf.placeholder(tf.float32, None, name='x_hat')
    noise_hat = tf.placeholder(tf.float32, None, name='noise_hat')
    y_hat = tf.reduce_sum(tf.multiply([-2.3, 4.5, 9.4], x_hat)) + noise_hat

    net = tf.reduce_sum(tf.multiply(x_hat, beta))

    delta_w = tf.multiply(tf.multiply(learning_rate, tf.subtract(y_hat, net, name='delta')), x_hat)

    learn = tf.assign(beta, tf.add(beta, delta_w))

# main function here

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

for i in range(1000):

    x = np.random.uniform(-10, 10, size=(2,))
    x = np.append(x, [1])
    noise = np.random.uniform(-1, 1)

    sess.run(learn, {x_hat: x, noise_hat: noise})

print sess.run(beta)

writer = tf.summary.FileWriter("path_to_folder", sess.graph)
writer.close()
