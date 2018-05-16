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
   parser.add_argument('--batch_size', required=False,default=32,type=int,help='Batch size')
   parser.add_argument('--l1_weight',  required=False,default=100.,type=float,help='Weight for L1 loss')
   parser.add_argument('--ig_weight',  required=False,default=1.,type=float,help='Weight for image gradient loss')
   parser.add_argument('--resBlocks',  required=False,default=4,type=int,help='Number of residual blocks in G')
   parser.add_argument('--network',    required=False,default='resnet',type=str,help='Network to use')
   parser.add_argument('--dataset',    required=False,default='gaze',type=str,help='Dataset to use')
   parser.add_argument('--epochs',     required=False,default=100,type=int,help='Number of epochs for GAN')
   a = parser.parse_args()

   batch_size    = a.batch_size
   l1_weight     = float(a.l1_weight)
   ig_weight     = float(a.ig_weight)
   network       = a.network
   dataset       = a.dataset
   epochs        = a.epochs
   resBlocks     = a.resBlocks
   
   experiment_dir  = 'checkpoints'\
                     +'/network_'+network\
                     +'/resBlocks_'+str(resBlocks)\
                     +'/l1_weight_'+str(l1_weight)\
                     +'/ig_weight_'+str(ig_weight)\
                     +'/dataset_'+dataset+'/'\

   images_dir      = experiment_dir+'images/'

   print
   print 'Creating',experiment_dir
   try: os.makedirs(images_dir)
   except: pass
   try: os.makedirs(TEST_images_dir)
   except: pass

   exp_info = dict()
   exp_info['batch_size']    = batch_size
   exp_info['l1_weight']     = l1_weight
   exp_info['ig_weight']     = ig_weight
   exp_info['network']       = network
   exp_info['dataset']          = dataset
   exp_info['epochs']        = epochs
   exp_pkl = open(experiment_dir+'info.pkl', 'wb')
   data = pickle.dumps(exp_info)
   exp_pkl.write(data)
   exp_pkl.close()

   print
   print 'batch_size:    ',batch_size
   print 'l1_weight:     ',l1_weight
   print 'ig_weight:     ',ig_weight
   print 'network:       ',network
   print 'dataset:       ',dataset
   print 'epochs:        ',epochs
   print

   if dataset == 'gaze':
      height   = 35
      width    = 55
      channels = 1

   if network == 'pix2pix': from pix2pix import *
   if network == 'resnet': from resnet import *

   # global step that is saved with a model to keep track of how many steps/epochs
   global_step = tf.Variable(0, name='global_step', trainable=False)

   # synthetic image
   image_s = tf.placeholder(tf.float32, shape=(batch_size, height, width, channels), name='image_s')

   # real image
   image_r = tf.placeholder(tf.float32, shape=(batch_size, height, width, channels), name='image_r')

   # generated real image (fake)
   gen_real = netG(image_s, resBlocks)

   # send real images to D
   D_real = netD(image_r)

   # send generated images to D
   D_fake = netD(gen_real, reuse=True)

   # cost functions
   errD = tf.reduce_mean(D_real) - tf.reduce_mean(D_fake)
   errG = -tf.reduce_mean(D_fake)

   # gradient penalty
   epsilon = tf.random_uniform([], 0.0, 1.0)
   x_hat = image_r*epsilon + (1-epsilon)*gen_real
   d_hat = netD(x_hat, reuse=True)
   gradients = tf.gradients(d_hat, x_hat)[0]
   slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1]))
   gradient_penalty = 10*tf.reduce_mean((slopes-1.0)**2)
   errD += gradient_penalty

   if l1_weight > 0.0:
      l1_loss = tf.reduce_mean(tf.abs(gen_real-image_r))
      errG += l1_weight*l1_loss

   if ig_weight > 0.0:
      ig_loss = loss_gradient_difference(image_r, gen_real)
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
      
   G_train_op = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(errG, var_list=g_vars, global_step=global_step)
   D_train_op = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(errD, var_list=d_vars)

   saver = tf.train.Saver(max_to_keep=1)

   init = tf.group(tf.local_variables_initializer(), tf.global_variables_initializer())
   sess = tf.Session()
   sess.run(init)

   # write out logs for tensorboard to the checkpointSdir
   summary_writer = tf.summary.FileWriter(experiment_dir+'/logs/', graph=tf.get_default_graph())

   tf.add_to_collection('vars', G_train_op)
   tf.add_to_collection('vars', D_train_op)

   ckpt = tf.train.get_checkpoint_state(experiment_dir)
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

   real_images, synthetic_images = data_ops.loadData(dataset)

   # just use the last 500 synthetic for testing
   train_synthetic, test_synthetic = synthetic_images[:10000], synthetic_images[10000:]

   real_len  = len(real_images)
   synth_len = len(train_synthetic)
   num_test  = len(test_synthetic)
   num_train = real_len + synth_len
   print 'train:',len(train_synthetic)
   print 'test:',len(test_synthetic)

   n_critic = 5

   epoch_num = step/(num_train/batch_size)

   while epoch_num < epochs:
      s = time.time()
      epoch_num = step/(num_train/batch_size)

      real_idx  = np.random.choice(np.arange(real_len), batch_size, replace=False)
      synth_idx = np.random.choice(np.arange(synth_len), batch_size, replace=False)
      batchReal  = real_images[real_idx]
      batchSynth = train_synthetic[synth_idx]

      for itr in xrange(n_critic):
         sess.run(D_train_op, feed_dict={image_s:batchSynth, image_r:batchReal})

      sess.run(G_train_op, feed_dict={image_s:batchSynth, image_r:batchReal})
      D_loss, G_loss, summary = sess.run([errD, errG, merged_summary_op], feed_dict={image_s:batchSynth, image_r:batchReal})

      summary_writer.add_summary(summary, step)

      ss = time.time()-s
      print 'epoch:',epoch_num,'step:',step,'D loss:',D_loss,'G_loss:',G_loss,'time:',ss
      step += 1
      
      if step%1 == 0:
         print 'Saving model...'
         saver.save(sess, experiment_dir+'checkpoint-'+str(step))
         saver.export_meta_graph(experiment_dir+'checkpoint-'+str(step)+'.meta')
         print 'Model saved\n'

         idx = np.random.choice(np.arange(num_test), batch_size, replace=False)
         batchSynth = test_synthetic[idx]

         print 'Testing...'
         gen_images = np.asarray(sess.run(gen_real, feed_dict={image_s:batchSynth}))

         c = 0
         for gen, real in zip(gen_images, batchSynth):
            misc.imsave(images_dir+str(step)+'_synth.png', real)
            misc.imsave(images_dir+str(step)+'_gen.png', gen)
            c += 1
            if c == 5: break
         print 'Done with test images'
