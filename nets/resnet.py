import tensorflow as tf
import tensorflow.contrib.layers as tcl
import sys

sys.path.insert(0, 'ops/')
from tf_ops import *

def resBlock(x, num):

   conv1 = tcl.conv2d(x, 64, 3, 1, activation_fn=tf.nn.relu, normalizer_fn=tcl.batch_norm, weights_initializer=tf.random_normal_initializer(stddev=0.02), scope='g_resconv1_'+str(num))
   print 'res_conv1:',conv1

   conv2 = tcl.conv2d(conv1, 64, 3, 1, activation_fn=tf.identity, normalizer_fn=tcl.batch_norm, weights_initializer=tf.random_normal_initializer(stddev=0.02), scope='g_resconv2_'+str(num))
   print 'res_conv2:',conv2
   
   output = tf.add(x,conv2)
   
   output = relu(output)

   print 'res_out:',output
   print
   return output


def netG(x, resBlocks):
   print 'x:',x
   res = resBlock(x, 0)
   for i in range(resBlocks-1):
      res = resBlock(res, i+1)

   conv1 = tcl.conv2d(res, 1, 1, 1, activation_fn=tf.nn.tanh, normalizer_fn=tcl.batch_norm, weights_initializer=tf.random_normal_initializer(stddev=0.02), scope='g_conv1')
   print 'conv1:',conv1
   return conv1


def netD(x, reuse=False):
   print
   print 'netD'
   
   sc = tf.get_variable_scope()
   with tf.variable_scope(sc, reuse=reuse):

      conv1 = tcl.conv2d(x, 64, 4, 2, activation_fn=tf.identity, weights_initializer=tf.random_normal_initializer(stddev=0.02), scope='d_conv1')
      conv1 = lrelu(conv1)
      
      conv2 = tcl.conv2d(conv1, 128, 4, 2, activation_fn=tf.identity, weights_initializer=tf.random_normal_initializer(stddev=0.02), scope='d_conv2')
      conv2 = lrelu(conv1)
      
      conv3 = tcl.conv2d(conv2, 256, 4, 2, activation_fn=tf.identity, weights_initializer=tf.random_normal_initializer(stddev=0.02), scope='d_conv3')
      conv3 = lrelu(conv3)
      
      conv4 = tcl.conv2d(conv3, 512, 4, 2, activation_fn=tf.identity, weights_initializer=tf.random_normal_initializer(stddev=0.02), scope='d_conv4')
      conv4 = lrelu(conv4)
      
      conv5 = tcl.conv2d(conv4, 1, 1, 1, activation_fn=tf.identity, weights_initializer=tf.random_normal_initializer(stddev=0.02), scope='d_conv5')

      print conv1
      print conv2
      print conv3
      print conv4
      print conv5
      return conv5

