#!/usr/bin/env python

import tensorflow as tf
import numpy as np



sess = tf.Session()

with tf.name_scope("name") as scope:
    x = tf.random_uniform((1, 1), -10, 10, dtype=tf.float32)
    noise = tf.random_uniform((1, 1), -1, 1, dtype=tf.float32)

    noisy_line = -6.7 * x + 2 + noise

for i in range(100):

    sess.run(noisy_line)
    print sess.run(x, noisy_line)

writer = tf.summary.FileWriter("path_to_folder", sess.graph)
writer.close()
