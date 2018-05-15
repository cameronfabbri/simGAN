'''

   Main training file

   The goal is to correct the colors in underwater images.
   CycleGAN was used to create images that appear to be underwater.
   Those will be sent into the generator, which will attempt to correct the
   colors.

'''

import cPickle as pickle
import tensorflow as tf
from scipy import misc
from tqdm import tqdm
import numpy as np
import argparse
import ntpath
import random
import glob
import time
import sys
import cv2
import os

# my imports
sys.path.insert(0, 'ops/')
sys.path.insert(0, 'nets/')
from tf_ops import *
import data_ops

if __name__ == '__main__':
   parser = argparse.ArgumentParser()
   parser.add_argument('--batch_size',    required=False,default=32,type=int,help='Batch size')
   parser.add_argument('--l1_weight',     required=False,default=100.,type=float,help='Weight for L1 loss')
   parser.add_argument('--ig_weight',     required=False,default=1.,type=float,help='Weight for image gradient loss')
   parser.add_argument('--network',       required=False,default='pix2pix',type=str,help='Network to use')
   parser.add_argument('--augment',       required=False,default=0,type=int,help='Augment data or not')
   parser.add_argument('--epochs',        required=False,default=100,type=int,help='Number of epochs for GAN')
   parser.add_argument('--data',          required=False,default='underwater_imagenet',type=str,help='Dataset to use')
   a = parser.parse_args()

   LEARNING_RATE = float(a.LEARNING_RATE)
   LOSS_METHOD   = a.LOSS_METHOD
   batch_size    = a.batch_size
   l1_weight     = float(a.l1_weight)
   ig_weight     = float(a.ig_weight)
   network       = a.network
   augment       = a.augment
   epochs        = a.epochs
   data          = a.data
   
   EXPERIMENT_DIR  = 'checkpoints/LOSS_METHOD_'+LOSS_METHOD\
                     +'/network_'+network\
                     +'/l1_weight_'+str(l1_weight)\
                     +'/ig_weight_'+str(ig_weight)\
                     +'/augment_'+str(augment)\
                     +'/data_'+data+'/'\

   IMAGES_DIR      = EXPERIMENT_DIR+'images/'

   print
   print 'Creating',EXPERIMENT_DIR
   try: os.makedirs(IMAGES_DIR)
   except: pass
   try: os.makedirs(TEST_IMAGES_DIR)
   except: pass

   # TODO add new things to pickle file - INCLUDING BATCH SIZE AND LEARNING RATE
   # write all this info to a pickle file in the experiments directory
   exp_info = dict()
   exp_info['LEARNING_RATE'] = LEARNING_RATE
   exp_info['LOSS_METHOD']   = LOSS_METHOD
   exp_info['batch_size']    = batch_size
   exp_info['l1_weight']     = l1_weight
   exp_info['ig_weight']     = ig_weight
   exp_info['network']       = network
   exp_info['augment']       = augment
   exp_info['epochs']        = epochs
   exp_info['data']          = data
   exp_pkl = open(EXPERIMENT_DIR+'info.pkl', 'wb')
   data = pickle.dumps(exp_info)
   exp_pkl.write(data)
   exp_pkl.close()
   
   print
   print 'LEARNING_RATE: ',LEARNING_RATE
   print 'LOSS_METHOD:   ',LOSS_METHOD
   print 'batch_size:    ',batch_size
   print 'l1_weight:     ',l1_weight
   print 'ig_weight:     ',ig_weight
   print 'network:       ',network
   print 'augment:       ',augment
   print 'epochs:        ',epochs
   print 'data:          ',data
   print

   if network == 'pix2pix': from pix2pix import *
   if network == 'resnet': from resnet import *

   # global step that is saved with a model to keep track of how many steps/epochs
   global_step = tf.Variable(0, name='global_step', trainable=False)

   # underwater image
   image_u = tf.placeholder(tf.float32, shape=(batch_size, 256, 256, 3), name='image_u')

   # correct image
   image_r = tf.placeholder(tf.float32, shape=(batch_size, 256, 256, 3), name='image_r')

   # generated corrected colors
   layers    = netG_encoder(image_u)
   gen_image = netG_decoder(layers)

   # send 'above' water images to D
   D_real = netD(image_r, LOSS_METHOD)

   # send corrected underwater images to D
   D_fake = netD(gen_image, LOSS_METHOD, reuse=True)

   e = 1e-12
   if LOSS_METHOD == 'least_squares':
      print 'Using least squares loss'
      errD_real = tf.nn.sigmoid(D_real)
      errD_fake = tf.nn.sigmoid(D_fake)
      errG = 0.5*(tf.reduce_mean(tf.square(errD_fake - 1)))
      errD = tf.reduce_mean(0.5*(tf.square(errD_real - 1)) + 0.5*(tf.square(errD_fake)))
   if LOSS_METHOD == 'gan':
      print 'Using original GAN loss'
      errD_real = tf.nn.sigmoid(D_real)
      errD_fake = tf.nn.sigmoid(D_fake)
      errG = tf.reduce_mean(-tf.log(errD_fake + e))
      errD = tf.reduce_mean(-(tf.log(errD_real+e)+tf.log(1-errD_fake+e)))
   if LOSS_METHOD == 'wgan':
      # cost functions
      errD = tf.reduce_mean(D_real) - tf.reduce_mean(D_fake)
      errG = -tf.reduce_mean(D_fake)

      # gradient penalty
      epsilon = tf.random_uniform([], 0.0, 1.0)
      x_hat = image_r*epsilon + (1-epsilon)*gen_image
      d_hat = netD(x_hat, LOSS_METHOD, reuse=True)
      gradients = tf.gradients(d_hat, x_hat)[0]
      slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1]))
      gradient_penalty = 10*tf.reduce_mean((slopes-1.0)**2)
      errD += gradient_penalty

   if l1_weight > 0.0:
      l1_loss = tf.reduce_mean(tf.abs(gen_image-image_r))
      errG += l1_weight*l1_loss

   if ig_weight > 0.0:
      ig_loss = loss_gradient_difference(image_r, image_u)
      errG += ig_weight*ig_loss

   # tensorboard summaries
   tf.summary.scalar('d_loss', tf.reduce_mean(errD))
   tf.summary.scalar('g_loss', tf.reduce_mean(errG))
   try: tf.summary.scalar('l1_loss', tf.reduce_mean(l1_loss))
   except: pass
   try: tf.summary.scalar('ig_loss', tf.reduce_mean(ig_loss))
   except: pass

   # get all trainable variables, and split by network G and network D
   t_vars = tf.trainable_variables()
   d_vars = [var for var in t_vars if 'd_' in var.name]
   g_vars = [var for var in t_vars if 'g_' in var.name]
      
   G_train_op = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(errG, var_list=g_vars, global_step=global_step)
   D_train_op = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(errD, var_list=d_vars)

   saver = tf.train.Saver(max_to_keep=2)

   init = tf.group(tf.local_variables_initializer(), tf.global_variables_initializer())
   sess = tf.Session()
   sess.run(init)

   # write out logs for tensorboard to the checkpointSdir
   summary_writer = tf.summary.FileWriter(EXPERIMENT_DIR+'/logs/', graph=tf.get_default_graph())

   tf.add_to_collection('vars', G_train_op)
   tf.add_to_collection('vars', D_train_op)

   ckpt = tf.train.get_checkpoint_state(EXPERIMENT_DIR)
   if ckpt and ckpt.model_checkpoint_path:
      print "Restoring previous model..."
      try:
         saver.restore(sess, ckpt.model_checkpoint_path)
         print "Model restored"
      except:
         print "Could not restore model"
         pass
   
   step = int(sess.run(global_step))

   merged_summary_op = tf.summary.merge_all()

   # underwater photos
   trainA_paths = np.asarray(glob.glob('datasets/'+data+'/trainA/*.jpg'))
   # normal photos (ground truth)
   trainB_paths = np.asarray(glob.glob('datasets/'+data+'/trainB/*.jpg'))
   # testing paths
   #test_paths = np.asarray(glob.glob('datasets/'+data+'/test/*.jpg'))
   test_paths = np.asarray(glob.glob('datasets/'+data+'/trainA/*.jpg'))

   print len(trainB_paths),'training images'

   num_train = len(trainB_paths)
   num_test  = len(test_paths)

   n_critic = 1
   if LOSS_METHOD == 'wgan': n_critic = 5

   epoch_num = step/(num_train/batch_size)

   # kernel for gaussian blurring
   kernel = np.ones((5,5),np.float32)/25

   while epoch_num < epochs:
      s = time.time()
      epoch_num = step/(num_train/batch_size)

      idx = np.random.choice(np.arange(num_train), batch_size, replace=False)
      batchA_paths = trainA_paths[idx]
      batchB_paths = trainB_paths[idx]
      
      batchA_images = np.empty((batch_size, 256, 256, 3), dtype=np.float32)
      batchB_images = np.empty((batch_size, 256, 256, 3), dtype=np.float32)

      i = 0
      for a,b in zip(batchA_paths, batchB_paths):
         a_img = data_ops.preprocess(misc.imread(a).astype('float32'))
         b_img = data_ops.preprocess(misc.imread(b).astype('float32'))

         # Data augmentation here - each has 50% chance
         if augment:
            r = random.random()
            # flip image left right
            if r < 0.5:
               #print 'lr'
               a_img = np.fliplr(a_img)
               b_img = np.fliplr(b_img)
            
            r = random.random()
            # flip image up down
            if r < 0.5:
               #print 'updown'
               a_img = np.flipud(a_img)
               b_img = np.flipud(b_img)
            
            r = random.random()
            # send in the clean image for both
            if r < 0.5:
               #print 'clean'
               a_img = b_img

            r = random.random()
            # perform some gaussian blur on distorted image
            if r < 0.5:
               #print 'blur'
               a_img = cv2.filter2D(a_img,-1,kernel)

         #misc.imsave('a_img.png', a_img)
         #misc.imsave('b_img.png', b_img)
         #exit()
         batchA_images[i, ...] = a_img
         batchB_images[i, ...] = b_img
         i += 1
      
      for itr in xrange(n_critic):
         sess.run(D_train_op, feed_dict={image_u:batchA_images, image_r:batchB_images})

      sess.run(G_train_op, feed_dict={image_u:batchA_images, image_r:batchB_images})
      D_loss, G_loss, summary = sess.run([errD, errG, merged_summary_op], feed_dict={image_u:batchA_images, image_r:batchB_images})

      summary_writer.add_summary(summary, step)

      ss = time.time()-s
      print 'epoch:',epoch_num,'step:',step,'D loss:',D_loss,'G_loss:',G_loss,'time:',ss
      step += 1
      
      if step%500 == 0:
         print 'Saving model...'
         saver.save(sess, EXPERIMENT_DIR+'checkpoint-'+str(step))
         saver.export_meta_graph(EXPERIMENT_DIR+'checkpoint-'+str(step)+'.meta')
         print 'Model saved\n'

         idx = np.random.choice(np.arange(num_test), batch_size, replace=False)
         batch_paths = test_paths[idx]
         
         batch_images = np.empty((batch_size, 256, 256, 3), dtype=np.float32)

         print 'Testing...'
         i = 0
         for a in batch_paths:
            a_img = misc.imread(a).astype('float32')
            a_img = data_ops.preprocess(misc.imresize(a_img, (256, 256, 3)))
            batch_images[i, ...] = a_img
            i += 1

         gen_images = np.asarray(sess.run(gen_image, feed_dict={image_u:batch_images}))

         c = 0
         for gen, real in zip(gen_images, batch_images):
            misc.imsave(IMAGES_DIR+str(step)+'_real.png', real)
            misc.imsave(IMAGES_DIR+str(step)+'_gen.png', gen)
            c += 1
            if c == 5: break
         print 'Done with test images'
