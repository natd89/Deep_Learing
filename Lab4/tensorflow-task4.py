
#!/usr/bin/env python

import tensorflow as tf
import numpy as np
import pandas


# tf stuff below
with tf.name_scope("name") as scope:

    beta = tf.Variable([0.0, 0.0, 0.0])

    learning_rate = tf.Variable(0.005)

    x_hat = tf.placeholder(tf.float32, None, name='x_hat')
    y_hat = tf.placeholder(tf.float32, None, name='y_hat')

    net = tf.reduce_sum(tf.multiply(x_hat, beta))

    delta_w = tf.multiply(tf.multiply(learning_rate, tf.subtract(y_hat, net, name='delta')), x_hat)

    learn = tf.assign(beta, tf.add(beta, delta_w))

# main function here

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

data = pandas.read_csv('./foo.csv', ',').as_matrix()
x = np.zeros((len(data),3))
x[:, 0:2] = data[:, 0:2][:, 0:2]
y = data[:, 2]

for i in range(len(y[:])):
    x[i, 2] = 1.0
    sess.run(learn, {x_hat: x[i, :], y_hat: y[i]})

print sess.run(beta)

writer = tf.summary.FileWriter("path_to_folder", sess.graph)
writer.close()
