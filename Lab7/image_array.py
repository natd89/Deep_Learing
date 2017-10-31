#!/usr/bin/env python


import tensorflow as tf
import numpy as np
import os
from skimage import data, io, filters, transform
import matplotlib.pyplot as plt
import random
from pdb import set_trace as brake
import math
import PIL

images = os.listdir('/home/nmd89/Pictures/celeb_images')




def fc(x, out_size=None, is_output=False, name="fc"):

    #x is an input tensor
    x_shape = x.get_shape().as_list()
    with tf.variable_scope(name):

        #  Create a W filter variable with the proper size
        W_fc = tf.get_variable(name+'_W', shape=[x_shape[1], out_size], dtype=tf.float32, initializer = tf.contrib.layers.variance_scaling_initializer())

        #  Create a B bias variable with the proper size
        B_fc = tf.get_variable(initializer = tf.random_normal([out_size]), name = name+'_B')

        #  Multiply x by W and add b
        out = tf.nn.bias_add(tf.matmul(x, W_fc), B_fc)

        #  If is_output is False,
        if not is_output:

            # Call the tf.nn.relu function
            out = tf.nn.relu(out, name = name+'relu_fc')

    return out





def conv( x, filter_size=3, stride=1, num_filters=64, is_output=False, name='conv', initializer=tf.contrib.layers.l2_regularizer(scale=0.1)):

    # x is an input tensor
    x_shape = x.get_shape().as_list()
    # Declare a name scope using the "name" parameter
    with tf.variable_scope('weights_'+name):

        # Create a W filter variable with the proper size
        W_conv = tf.get_variable(name+'_W', shape = [filter_size,filter_size,x_shape[3],num_filters], dtype=tf.float32, initializer = tf.contrib.layers.variance_scaling_initializer())
        
        # Create a B bias variable withthe proper size
        B_conv = tf.Variable(tf.random_normal([num_filters]), name = name+'_B')

        # Convolve x with W by calling the tf.nn.conv2d function
        convolve = tf.nn.conv2d(x,W_conv, strides = [1, stride, stride, 1], padding='SAME')
    with tf.name_scope(name):
        # Add the bias
        convolve_bias = tf.nn.bias_add(convolve,B_conv)
        # If is_output is False call the tf.nn.relu function
        # if not is_output:
        convolve_bias = tf.nn.relu(convolve_bias, name=name+'relu_conv')
        # convolve_bias = tf.maximum(convolve_bias, 0.2*convolve_bias)

    return convolve_bias




def up_conv(x, name='upconv', filter_size=3, num_filters=64, stride=2, is_output=False):

    x_shape = x.get_shape().as_list()
    with tf.variable_scope('weights_'+name):
        
        W_up = tf.get_variable(name+'_W_up', shape = [filter_size,filter_size,num_filters,x_shape[3]], dtype = tf.float32, initializer = tf.contrib.layers.variance_scaling_initializer()) 

        out_shape = [x_shape[0],x_shape[1]*2,x_shape[2]*2,num_filters]

        if not is_output:
            h = tf.contrib.layers.layer_norm(x, center=True)
        else:
            h = x

        transpose = tf.nn.conv2d_transpose(h, W_up, out_shape, [1,stride,stride,1], name = name)

    with tf.name_scope(name):

        if not is_output:
            up_con = tf.nn.relu(transpose, name=name+'relu_upconv')
        if is_output:
            up_con = tf.nn.tanh(transpose, name=name+'_tanh')

    return transpose







def discriminator( x, reuse=False, name='discriminator'):

    x_shape = tf.shape(x)
    batch_size = x_shape[0]
    
    if reuse:
        tf.get_variable_scope().reuse_variables()    

    with tf.name_scope(name):
       
        # h0 = conv( x, filter_size=3, stride=2, num_filters=64, name='disc_1') 
        h1 = conv( x, filter_size=3, stride=2, num_filters=128, name='disc_2') 
        h2 = conv( h1, filter_size=3, stride=2, num_filters=256, name='disc_3')
        h3 = conv( h2, filter_size=3, stride=2, num_filters=512, name='disc_4')
        h4 = conv( h3, filter_size=3, stride=2, num_filters=1024, name='disc_5')
        h5 = tf.reshape( h4, [batch_size,4*4*1024])
        h6 = fc( h5, out_size=1, is_output=True, name='disc_fc')
    
    return h6





def generator( z, reuse=False, name='generator' ):
  
    z_shape = tf.shape(z)
    batch_size = z_shape[0]

    if reuse:
        tf.get_variable_scope().reuse_variables()

    with tf.name_scope(name):
        
        fc_out = fc(z, out_size=4*4*1024, name='gen_fc')
        h0 = tf.reshape(fc_out, [batch_size, 4, 4, 1024], name = 'gen_reshape')
        h1 = up_conv(h0, name='gen_1', num_filters=512)
        h2 = up_conv(h1, name='gen_2', num_filters = 256)
        h3 = up_conv(h2, name='gen_3', num_filters = 128)
        h4 = up_conv(h3, name='gen_4', num_filters = 3)
        # h4 = up_conv(h3, name='gen_4', num_filters = 64)
        # h5 = up_conv(h4, name='gen_5', num_filters = 3)

    return h4



def slerp(p0, p1, t):
    omega = np.arccos(np.dot(p0/np.linalg.norm(p0), p1/np.linalg.norm(p1)))
    so = np.sin(omega)
    return np.sin((1.0-t)*omega)/so*p0+np.sin(t*omega)/so*p1



lamb=10
n_critic=5
alpha=0.0001
beta_1=0.0
beta_2 = 0.9
im_size = 64



z = tf.placeholder(tf.float32, shape=[64, 100], name='z')
image = tf.placeholder(tf.float32, shape=[64, im_size, im_size, 3], name='image')
eps = tf.placeholder(tf.float32, shape = [64, 1, 1, 1], name='eps')




with tf.variable_scope('Model'):
    
    # get the generated image
    G = generator(z)
        
    # get discriminator value
    dx = discriminator(image, name='input_discriminator')
    
    x_hat = eps*image + (1-eps)*G
    
    d_xtilde = discriminator(G, name='x_tilde_discriminator', reuse=True)
    d_xhat   = discriminator(x_hat, name='x_hat_discriminator', reuse=True)
    
    with tf.name_scope('Loss'):
        # compute the loss function
        L = d_xtilde - dx + lamb * ( tf.norm( tf.gradients( d_xhat, x_hat ) , 2 ) -1 )**2

    # get variables to train 
    vars = tf.trainable_variables()
    disc_params = [v for v in vars if 'disc' in v.name]
    gen_params  = [v for v in vars if 'gen' in v.name]
        
    

    
with tf.name_scope('Optimizers'):
     
    disc_weights = tf.train.AdamOptimizer(alpha, beta_1, beta_2).minimize(tf.reduce_mean(L), var_list=[disc_params])
      
    gen_weights  = tf.train.AdamOptimizer(alpha, beta_1, beta_2).minimize(tf.reduce_mean(-d_xtilde), var_list=[gen_params])
    tf.add_to_collection('disc_weights', disc_weights)
    tf.add_to_collection('gen_weights', gen_weights)





sess = tf.Session()
saver = tf.train.Saver()
saver.restore(sess, './GAN_data/GAN.ckpt')

init = tf.global_variables_initializer()
sess.run(init)


batch_size = 64

side = int(np.sqrt(batch_size))

z_ = np.random.uniform(size=(batch_size, 100))
im_array = sess.run(G, feed_dict={z: z_})
clip_neg = im_array<0
im_array[clip_neg] = 0.0
fig2 = plt.figure()

for i in range(batch_size):
    fig2.add_subplot(side,side, i+1)
    plt.imshow(im_array[i,:,:,:])
    plt.axis('off')

plt.show()

