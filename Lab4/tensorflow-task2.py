#!/usr/bin/env python

import tensorflow as tf
import numpy as np
import pdb

# tf stuff below
with tf.name_scope("name") as scope:

    x_hat = tf.placeholder(tf.float32, None, name='x_hat')
    noise_hat = tf.placeholder(tf.float32, None, name='noise_hat')
    y_hat = -6.7 * x_hat + 2 + noise_hat

    m = tf.Variable([0.1])
    b = tf.Variable([0.1])

    delta = tf.subtract(y_hat, tf.multiply(m, x_hat)+b, name='delta')

    learn_m = tf.assign(m, tf.add(m, tf.multiply(delta, x_hat)))
    learn_b = tf.assign(b, tf.add(b, tf.multiply(delta, 1.0)))

# main function here

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

for i in range(100):

    x = np.random.uniform(-10, 10)
    noise = np.random.uniform(-1, 1)

    sess.run([learn_m, learn_b], {x_hat: x, noise_hat: noise})
