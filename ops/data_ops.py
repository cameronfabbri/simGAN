'''

Operations used for data management

MASSIVE help from https://github.com/affinelayer/pix2pix-tensorflow/blob/master/pix2pix.py

'''

from __future__ import division
from __future__ import absolute_import

from tqdm import tqdm
from scipy import misc
from skimage import color
import tensorflow as tf
import numpy as np
import random
import glob
import os
import fnmatch
import cv2


# [-1,1] -> [0, 255]
def deprocess(x):
   return (x+1.0)*127.5

# [0,255] -> [-1, 1]
def preprocess(x,min_=-1,max_=1):
#   return (x/127.5)-1.0
   t = x-np.min(x)
   b = np.max(x)-np.min(x)
   x = (max_-min_)*(t/b)+min_
   return x

def getFeedDict(utrain_paths,places2_paths,BATCH_SIZE):

   utrain_batch    = random.sample(utrain_paths, BATCH_SIZE)
   places2train_batch = random.sample(places2_paths, BATCH_SIZE)

   img = tf.image.decode_image(utrain_batch[0])
   print img


   exit()

def getPaths(data_dir,ext='png'):
   pattern   = '*.'+ext
   image_paths = []
   for d, s, fList in os.walk(data_dir):
      for filename in fList:
         if fnmatch.fnmatch(filename, pattern):
            fname_ = os.path.join(d,filename)
            image_paths.append(fname_)
   return image_paths


def loadData(dataset, train=True):

   if dataset == 'gaze':
      synthetic_paths = getPaths('/mnt/data1/images/eye_gaze/SynthEyes_data/')
      real_paths      = getPaths('/mnt/data1/images/eye_gaze/MPIIGaze/images/')

   num_s = len(synthetic_paths)
   num_r = len(real_paths)

   synthetic_images = np.zeros((num_s, 35, 55, 1), dtype=np.float32)
   real_images = np.zeros((num_r, 35, 55, 1), dtype=np.float32)

   print 'Loading synthetic images...'
   i = 0
   for s in tqdm(synthetic_paths):
      img = misc.imread(s)
      img = misc.imresize(img, (35, 55))
      img = color.rgb2gray(img)
      img = np.expand_dims(img, 2)
      # normalize
      img = preprocess(img)
      synthetic_images[i, ...] = img
      i += 1
   
   print 'Loading real images...'
   i = 0
   for s in tqdm(real_paths):
      img = misc.imread(s)
      img = misc.imresize(img, (35, 55))
      img = color.rgb2gray(img)
      img = np.expand_dims(img, 2)
      # normalize
      img = preprocess(img)
      real_images[i, ...] = img
      i += 1

   return real_images, synthetic_images

