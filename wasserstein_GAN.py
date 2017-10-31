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
    
    if reuse:
        tf.get_variable_scope().reuse_variables()    

    with tf.name_scope(name):
        
        h0 = conv( x, filter_size=3, stride=2, num_filters=64, name='disc_1') 
        h1 = conv( h0, filter_size=3, stride=2, num_filters=128, name='disc_2') 
        h2 = conv( h1, filter_size=3, stride=2, num_filters=256, name='disc_3')
        h3 = conv( h2, filter_size=3, stride=2, num_filters=512, name='disc_4')
        h4 = conv( h3, filter_size=3, stride=2, num_filters=1024, name='disc_5')
        h5 = tf.reshape( h4, [batch_size,4*4*1024])
        h6 = fc( h5, out_size=1, is_output=True, name='disc_fc')
    
    return h6



# ------------------------Function for the Generator---------------------------- #

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
        # h4 = up_conv(h3, name='gen_4', num_filters = 3)
        h4 = up_conv(h3, name='gen_4', num_filters = 64)
        h5 = up_conv(h4, name='gen_5', num_filters = 3)

    return h5




images = os.listdir('/home/nmd89/Pictures/celeb_images')

batch_size = 1


lamb=10
n_critic=5
alpha=0.0001
beta_1=0.0
beta_2 = 0.9
im_size = 128


z = tf.placeholder(tf.float32, shape=[batch_size, 100], name='z')
image = tf.placeholder(tf.float32, shape=[batch_size, im_size, im_size, 3], name='image')
eps = tf.placeholder(tf.float32, shape = [batch_size, 1, 1, 1], name='eps')




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




with tf.Session() as sess:

    init = tf.global_variables_initializer()
    sess.run(init)
    saver = tf.train.Saver()
   

    # summaries
    writer = tf.summary.FileWriter('./tf_logs', sess.graph)
    gen_im = tf.summary.image('generated_image', G, max_outputs = batch_size)
    input_im = tf.summary.image('input_image', image, max_outputs = batch_size)
    merge = tf.summary.merge_all()

    # run the training loop
    for i in range(100000):

        for j in range(n_critic):
            
            rand_index = random.sample(range(200000), batch_size)
            rand_index = [300]

            z_input = np.random.uniform(size=(batch_size, 100))

            loaded_images = np.zeros((batch_size, 128, 128, 3))
            resized_images = np.zeros((batch_size, im_size, im_size, 3))

            for k in range(batch_size):
                # loaded_images[k,:,:,:] = io.imread('/home/nmd89/Pictures/celeb_images/'+images[rand_index[k]])
                loaded_images[k,:,:,:] = io.imread('/home/nmd89/Pictures/celeb_images/'+images[rand_index[k]]) 
                # resized_images[k,:,:,:] = transform.resize(loaded_images[k,:,:,:], ( im_size, im_size, 3))

            epsilon = np.random.uniform(size=(batch_size, 1, 1, 1))
            
            disc_opt, ss = sess.run([disc_weights, merge], feed_dict={z: z_input, image: loaded_images, eps: epsilon})
            # disc_opt, ss = sess.run([disc_weights, merge], feed_dict={z: z_input, image: resized_images, eps: epsilon})

        gen_opt, ss =sess.run([gen_weights, merge], feed_dict={z: z_input, image: loaded_images})
        # gen_opt, ss =sess.run([gen_weights, merge], feed_dict={z: z_input, image: resized_images})
        writer.add_summary(ss, i*n_critic + j)
        saver.save(sess, "GAN_data/GAN.ckpt")
        print 'iteration ' + str(i) + ' complete'

writer.close()

