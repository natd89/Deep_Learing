
#!/usr/bin/env python

import tensorflow as tf
import numpy as np
import os
from skimage import data, io, filters, transform
import random
from pdb import set_trace as brake
import math

tf.reset_default_graph()

n = 150
n_test = 75

batch_size = 10

pos_test = os.listdir('/home/nmd89/Downloads/cancer_data/inputs/pos_test')
pos_train = os.listdir('/home/nmd89/Downloads/cancer_data/inputs/pos_train')


pos_test_lab = os.listdir('/home/nmd89/Downloads/cancer_data/outputs/pos_test')
pos_train_lab = os.listdir('/home/nmd89/Downloads/cancer_data/outputs/pos_train')


# select n random images to train with
index = random.sample(range(len(pos_train)), n)


# load the training data into a tensor
im_size = 64
image_train = np.zeros((n, im_size, im_size, 3)).astype(np.float32)
image_train_lab = np.zeros((n, im_size, im_size)).astype(np.float32)



for i in range(n):
  
    im_pos_tr = io.imread('/home/nmd89/Downloads/cancer_data/inputs/pos_train/'+pos_train[index[i]])

    im_pos_lab= io.imread('/home/nmd89/Downloads/cancer_data/outputs/pos_train/'+pos_train_lab[index[i]])    


    print i
    
    image_train[i, :, :, :] = transform.resize(im_pos_tr, (im_size, im_size, 3))
    image_train_lab[i, :,:] = transform.resize(im_pos_lab, (im_size, im_size))
    


# whiten the data
image_train = (image_train - np.mean(image_train,0))/(np.std(image_train,0))




image_test = np.zeros((len(pos_test),im_size, im_size, 3)).astype(np.float32)
image_test_lab = np.zeros((len(pos_test),im_size, im_size)).astype(np.float32)

# load the test data into a tensor
for i in range(n_test):

    if i==0:
        im_pos_tst = io.imread('/home/nmd89/Downloads/cancer_data/inputs/pos_test/pos_test_000072.png')
        im_pos_tst_lab = io.imread('/home/nmd89/Downloads/cancer_data/outputs/pos_test/pos_test_000072.png')

    else:
        im_pos_tst = io.imread('/home/nmd89/Downloads/cancer_data/inputs/pos_test/'+pos_test[i])
        im_pos_tst_lab = io.imread('/home/nmd89/Downloads/cancer_data/outputs/pos_test/'+pos_test_lab[i])

    image_test[i,:,:,:] = transform.resize(im_pos_tst, (im_size, im_size, 3))
    image_test_lab[i,:,:] = transform.resize(im_pos_tst_lab, (im_size, im_size))

    print i

# whiten the test images
image_test = (image_test - np.mean(image_test,0))/(np.std(image_test,0))



def conv( x, filter_size=3, stride=1, num_filters=64, is_output=False, name='conv', initializer=tf.contrib.layers.l2_regularizer(scale=0.1)):

    # x is an input tensor
    x_shape = x.get_shape().as_list()
    # Declare a name scope using the "name" parameter
    with tf.variable_scope(name):

        # Create a W filter variable with the proper size
        W_conv = tf.get_variable('W' + name, shape = [filter_size,filter_size,x_shape[3],num_filters], dtype=tf.float32, initializer = tf.contrib.layers.variance_scaling_initializer())
        
        # Create a B bias variable withthe proper size
        B_conv = tf.Variable(tf.random_normal([num_filters]), name = 'B'+name)

        # Convolve x with W by calling the tf.nn.conv2d function
        convolve = tf.nn.conv2d(x,W_conv, strides = [1, stride, stride, 1], padding='SAME')

        # Add the bias
        convolve_bias = tf.nn.bias_add(convolve,B_conv)
        # If is_output is False call the tf.nn.relu function
        if not is_output:
            convolve_bias = tf.nn.relu(convolve_bias, name='relu')

    return convolve_bias



def max_pool(x, name):
    with tf.variable_scope(name):
        pooled = tf.nn.max_pool(x, ksize = [1,2,2,1], strides = [1,2,2,1], padding = 'SAME', name = name + 'pooled')
    return pooled



def conv_trans(x, name='upconv', filter_size=3, num_filters=64, stride=1):

    x_shape = x.get_shape().as_list()
    with tf.variable_scope(name):

        W_up = tf.get_variable('W_up'+name, shape = [filter_size,filter_size,num_filters,x_shape[3]], dtype = tf.float32, initializer = tf.contrib.layers.variance_scaling_initializer()) 

        out_shape = [x_shape[0],x_shape[1]*stride,x_shape[2]*stride,num_filters]

        transpose = tf.nn.conv2d_transpose(x, W_up, out_shape, [1,stride,stride,1], name = name + 'upconv')

    return transpose
 


def calc_accuracy(logits, label):
    
    accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(logits,3),tf.cast(label,tf.int64)),tf.float32))

    return accuracy


data = tf.placeholder(tf.float32, shape = [batch_size, im_size, im_size, 3], name = 'input')
labels = tf.placeholder(tf.float32, shape = [batch_size, im_size, im_size], name = 'label')
prob = tf.placeholder(tf.float32, shape = [], name = 'drop_prob')

h0 = conv(data, name='h0_')
h1 = conv(h0, name='h1_')

mp0 = max_pool(h1, 'mp0_')
h2 = conv(mp0, num_filters = 128, name ='h2_')
drop_1 = tf.nn.dropout(h2, keep_prob=prob, name='drop_out_1')
h3 = conv(drop_1, num_filters = 128, name = 'h3_')

mp1 = max_pool(h3, 'mp3_')
h4 = conv(mp1, num_filters = 256, name = 'h4_')
h5 = conv(h4, num_filters = 256, name = 'h5_')
u0 = conv_trans(h5, stride=2, num_filters=128, name='u0_')

c0 = tf.concat([h3,u0], axis=3, name='concat_0')
h6 = conv(c0, num_filters=128, name='h6_')
drop_2 = tf.nn.dropout(h6, keep_prob=prob, name='drop_out_2')
h7 = conv(drop_2, num_filters=128, name='h7_')

u1 = conv_trans(h7, stride=2, num_filters=64, name='u1_')
c1 = tf.concat([u1,h1],axis=3, name='concat_1')
h8 = conv(c1, num_filters=64, name='h8_')
h9 = conv(h8, num_filters=64, name='h9_')
h10 = conv(h9, num_filters=2, name='h10_')

with tf.name_scope('loss'):
    vars = tf.trainable_variables()
    lossL2 = tf.add_n([tf.nn.l2_loss(v) for v in vars])*0.001
    loss = tf.add(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.cast(labels,tf.int32), logits=h10), lossL2)

    # loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.cast(labels,tf.int32), logits=h10)

accuracy = calc_accuracy(h10, labels)


train = tf.train.AdamOptimizer(1e-4).minimize(loss)


with tf.Session() as sess:
    
    init = tf.global_variables_initializer()
    sess.run(init)
    writer = tf.summary.FileWriter('./tf_logs', sess.graph)
    lss = tf.summary.scalar('loss', tf.reduce_mean(loss))
    acc = tf.summary.scalar('accuracy', accuracy)
    out_msk = tf.summary.image('out_mask', tf.reshape(tf.cast(tf.argmax(h10,3),tf.float32),[batch_size,im_size,im_size,1]), max_outputs=batch_size)
    in_im = tf.summary.image('input_image',tf.reshape(tf.cast(data,tf.float32),[batch_size,im_size,im_size,3]), max_outputs=batch_size)
    in_msk = tf.summary.image('input_mask',tf.reshape(tf.cast(labels,tf.float32),[batch_size,im_size,im_size,1]), max_outputs=batch_size)
    merged = tf.summary.merge([lss, acc, out_msk, in_im, in_msk])

    in_im_tst = tf.summary.image('input_test_image',tf.reshape(tf.cast(data,tf.float32),[batch_size,im_size,im_size,3]), max_outputs=batch_size)
    in_msk_tst = tf.summary.image('input_test_mask',tf.reshape(tf.cast(labels,tf.float32),[batch_size,im_size,im_size,1]), max_outputs=batch_size)
    out_msk_tst = tf.summary.image('out_mask_tst', tf.reshape(tf.cast(tf.argmax(h10,3),tf.float32),[batch_size,im_size,im_size,1]), max_outputs=batch_size)
    acc_tst = tf.summary.scalar('test_accuracy', accuracy)
    merged_test = tf.summary.merge([acc_tst,in_im_tst, in_msk_tst,out_msk_tst])

    for k in range(10):
        print k
        i = 0
        # for i in range(n-batch_size):
        while i < (n-batch_size):
            print i        
            rand_index = random.sample(range(n), batch_size)
            
            sess.run(train, feed_dict = {data: np.reshape(image_train[rand_index,:,:,:], [batch_size,im_size,im_size,3]), labels: np.reshape(image_train_lab[rand_index,:,:], [batch_size,im_size,im_size]), prob: 0.6})
        
            summ = sess.run(merged, feed_dict={data: np.reshape(image_train[rand_index,:,:,:], [batch_size,im_size,im_size,3]), labels: np.reshape(image_train_lab[rand_index,:,:], [batch_size,im_size,im_size]), prob: 1})
            writer.add_summary(summ)

            i = i + batch_size
        
            # sess.run(train, feed_dict={data: np.reshape(image_train[0:batch_size,:,:,:], [batch_size,im_size,im_size,3]), labels: np.reshape(image_train_lab[0:batch_size,:,:], [batch_size,im_size,im_size])})
            
            # summ_train = sess.run(merged, feed_dict={data: np.reshape(image_train[0:batch_size,:,:,:], [batch_size,im_size,im_size,3]), labels: np.reshape(image_train_lab[0:batch_size,:,:], [batch_size,im_size,im_size])})
            # writer.add_summary(summ_train)
        
            if n%10==0:
                
                rand_index_tst = random.sample(range(n_test), batch_size-1)

                rand_index_tst.insert(0, 0)
                
                summ_test = sess.run(merged_test, feed_dict={data: np.reshape(image_test[rand_index_tst,:,:,:], [batch_size,im_size,im_size,3]).astype(np.float32), labels: np.reshape(image_test_lab[rand_index_tst,:,:], [batch_size,im_size,im_size]).astype(np.float32), prob: 1})

                writer.add_summary(summ_test)

writer.close()

