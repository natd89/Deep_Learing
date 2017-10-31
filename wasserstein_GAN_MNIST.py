#!/usr/bin/env python

from NN_Functions import fc, conv, up_conv
import tensorflow as tf
import numpy as np
import os
from skimage import data, io, filters, transform
import random
from pdb import set_trace as brake
import math
import PIL

tf.reset_default_graph()


# ------------------------Function for the Discriminator---------------------------- #

def discriminator( x, reuse=False, name='discriminator'):

    x_shape = tf.shape(x)
    batch_size = x_shape[0]
    
    with tf.name_scope(name):

        if reuse:
            tf.get_variable_scope().reuse_variables()    
        
        h0 = conv( x, filter_size=3, stride=2, num_filters=64, name='disc_1') 
        h1 = conv( h0, filter_size=3, stride=2, num_filters=128, name='disc_2') 
        h2 = tf.reshape( h1, [batch_size,7*7*128])
        h3 = fc( h2, out_size=1, is_output=True, name='disc_fc')
    
    return h3



# ------------------------Function for the Generator---------------------------- #

def generator( z, reuse=False, name='generator' ):
  
    z_shape = tf.shape(z)
    batch_size = z_shape[0]

    with tf.name_scope(name):

        if reuse:
            tf.get_variable_scope().reuse_variables()
        
        fc_out = fc(z, out_size=7*7*128, name='gen_fc')
        h0 = tf.reshape(fc_out, [batch_size, 7, 7, 128], name = 'gen_reshape')
        h1 = up_conv(h0, name='gen_1', num_filters=64)
        h2 = up_conv(h1, name='gen_2', num_filters = 1, is_output=True)
      
    return h2




images = os.listdir('/home/nathan/Pictures/mnist_png/training')
print len(images)

batch_size = 1


lamb=10
n_critic=1
alpha=0.0002
beta_1=0.5
beta_2 = 0.999
im_size = 28


z = tf.placeholder(tf.float32, shape=[batch_size, 100], name='z')
image = tf.placeholder(tf.float32, shape=[batch_size, im_size, im_size, 1], name='image')
eps = tf.placeholder(tf.float32, shape = [batch_size, 1, 1, 1], name='eps')




with tf.variable_scope('Model'):
    
    # get the generated image
    
    G = generator(z)
    
    # get discriminator value
    dx = discriminator(image, name='input_discriminator')
    
    x_hat = eps*image + (1-eps)*G
    
    d_xtilde = discriminator(G, name='x_tilde_discriminator', reuse=True)
    d_xhat   = discriminator(x_hat, name='x_hat_discriminator', reuse=True)
    gradients = tf.gradients( d_xhat, x_hat )

    with tf.name_scope('Loss'):
        # XXX is the use of tf.norm here correct?
        # compute the loss function
        L = tf.reduce_mean(d_xtilde - dx + lamb * ( tf.norm( gradients) -1 )**2)

    # get variables to train 
    vars = tf.trainable_variables()
    disc_params = [v for v in vars if 'disc' in v.name]
    gen_params  = [v for v in vars if 'gen' in v.name]
        
    

    
with tf.name_scope('Optimizers'):
     
    disc_weights = tf.train.AdamOptimizer(alpha, beta1=beta_1, beta2=beta_2).minimize(L, var_list=disc_params)
      
    gen_weights  = tf.train.AdamOptimizer(alpha, beta1=beta_1, beta2=beta_2).minimize(tf.reduce_mean(-d_xtilde), var_list=gen_params)

    tf.add_to_collection('disc_weights', disc_weights)
    tf.add_to_collection('gen_weights', gen_weights)




with tf.Session() as sess:

    init = tf.global_variables_initializer()
    sess.run(init)
    saver = tf.train.Saver()
   

    # summaries
    writer = tf.summary.FileWriter('./tf_logs', sess.graph)
    gen_im = tf.summary.image('generated_image', G, max_outputs = batch_size)
    input_im = tf.summary.image('input_image', image, max_outputs = batch_size)
    lss    =  tf.summary.scalar('loss', L)
    lss_gen = tf.summary.scalar('gen_loss', tf.reduce_mean(-d_xtilde))
    grad = tf.summary.scalar('gradients', tf.reduce_mean(gradients))
    merge = tf.summary.merge_all()

    z_input = np.random.uniform(size=(batch_size, 100))

    # run the training loop
    for i in range(60000):

        for j in range(n_critic):
            
            rand_index = random.sample(range(60000), batch_size)
         
            rand_index = [1]

            # z_input = np.random.uniform(size=(batch_size, 100))

            loaded_images = np.zeros((batch_size, 28, 28, 1))

            for k in range(batch_size):

                loaded_images[k,:,:,:] = np.reshape(io.imread('/home/nathan/Pictures/mnist_png/training/'+images[rand_index[k]]), [im_size,im_size,1]) 


            epsilon = np.random.uniform(size=(batch_size, 1, 1, 1))
            
            disc_opt = sess.run(disc_weights, feed_dict={z: z_input, image: loaded_images, eps: epsilon})
                        
        epsilon = np.random.uniform(size=(batch_size, 1, 1, 1))
        gen_opt =sess.run(gen_weights, feed_dict={z: z_input, image: loaded_images, eps: epsilon})


        ss = sess.run(merge, feed_dict={z: z_input, image: loaded_images, eps: epsilon})
    
        writer.add_summary(ss, i*n_critic + j)
        saver.save(sess, "GAN_data/GAN.ckpt")
        print 'iteration ' + str(i) + ' complete'

writer.close()

