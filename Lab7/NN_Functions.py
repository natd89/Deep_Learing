#!/usr/bin/env python

import tensorflow as tf
import numpy as np
import os
from skimage import data, io, filters, transform
import random
from pdb import set_trace as brake
import math
import PIL




# ----------------------------Fully Connected Function-------------------------------- #

def fc(x, out_size=None, is_output=False, name="fc"): 

    #x is an input tensor
    x_shape = x.get_shape().as_list()
    with tf.variable_scope(name):

        #  Create a W filter variable with the proper size
        W_fc = tf.get_variable(name+'_W', shape=[x_shape[1], out_size], dtype=tf.float32, initializer = tf.contrib.layers.variance_scaling_initializer())

        #  Create a B bias variable with the proper size
        B_fc = tf.get_variable(name = name+'_B', shape=[out_size], dtype=tf.float32, initializer = tf.contrib.layers.variance_scaling_initializer())
        #  Multiply x by W and add b
        out = tf.nn.bias_add(tf.matmul(x, W_fc), B_fc)

        #  If is_output is False,
        if not is_output:

            # Call the tf.nn.relu function
            out = tf.nn.relu(out, name = name+'relu_fc')

    return out



# ----------------------------Convolution Function-------------------------------- #

def conv( x, filter_size=3, stride=1, num_filters=64, is_output=False, name='conv', initializer=tf.contrib.layers.l2_regularizer(scale=0.1)):

    # x is an input tensor
    x_shape = x.get_shape().as_list()
    # Declare a name scope using the "name" parameter
    with tf.variable_scope('weights_'+name):

        # Create a W filter variable with the proper size
        W_conv = tf.get_variable(name+'_W', shape = [filter_size,filter_size,x_shape[3],num_filters], dtype=tf.float32, initializer = tf.contrib.layers.variance_scaling_initializer())
        
        # Create a B bias variable withthe proper size
        B_conv = tf.get_variable(name = name+'_B', shape=[num_filters], dtype=tf.float32, initializer=tf.contrib.layers.variance_scaling_initializer())

        # Convolve x with W by calling the tf.nn.conv2d function
        convolve = tf.nn.conv2d(x,W_conv, strides = [1, stride, stride, 1], padding='SAME')

    with tf.name_scope(name):
        # Add the bias
        convolve_bias = tf.nn.bias_add(convolve,B_conv)
        # If is_output is False call the tf.nn.relu function
        if not is_output:
            convolve_bias = tf.nn.relu(convolve_bias, name=name+'relu_conv')
        # convolve_bias = tf.maximum(convolve_bias, 0.2*convolve_bias)

    return convolve_bias





# ----------------------------Transpose Convolution Function-------------------------------- #

def up_conv(x, name='upconv', filter_size=3, num_filters=64, stride=2, is_output=False):

    x_shape = x.get_shape().as_list()
    with tf.variable_scope('weights_'+name):

        W_up = tf.get_variable(name+'_W_up', shape = [filter_size,filter_size,num_filters,x_shape[3]], dtype = tf.float32, initializer = tf.contrib.layers.variance_scaling_initializer()) 

        out_shape = [x_shape[0],x_shape[1]*stride,x_shape[2]*stride,num_filters]

        
        B_up = tf.get_variable(name = name+'_B_up', shape=[num_filters], dtype=tf.float32, initializer=tf.contrib.layers.variance_scaling_initializer())
        
        
        # if not is_output:
            # h = tf.contrib.layers.layer_norm(x, center=True)
        # else:
        h = x

        transpose = tf.nn.conv2d_transpose(h, W_up, out_shape, [1,stride,stride,1], name = name)

        transpose_bias = tf.nn.bias_add(transpose, B_up)
       

    with tf.name_scope(name):

        if not is_output:
            up_conv = tf.nn.relu(transpose_bias, name=name+'relu_upconv')
        if is_output:
            up_conv = tf.nn.tanh(transpose_bias, name=name+'_tanh')
            # up_conv = transpose_bias

    return up_conv


