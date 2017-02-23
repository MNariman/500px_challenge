from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np
import argparse
import sys

from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf

FLAGS = None


def main(_):
  # Import data
  mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)

  # Create the model
  x = tf.placeholder(tf.float32, [None, 784])
  
  

  W = tf.Variable(tf.zeros([784, 10]))
  b = tf.Variable(tf.zeros([10]))
  y = tf.matmul(x, W) + b

  # Define loss and optimizer
  y_ = tf.placeholder(tf.float32, [None, 10])

  # The raw formulation of cross-entropy,
  #
  #   tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(tf.nn.softmax(y)),
  #                                 reduction_indices=[1]))
  #
  # can be numerically unstable.
  #
  # So here we use tf.nn.softmax_cross_entropy_with_logits on the raw
  # outputs of 'y', and then average across the batch.
  cross_entropy = tf.reduce_mean(
      tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
  
  train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

  sess = tf.InteractiveSession()
  tf.global_variables_initializer().run()
  

  # Train The Neural Network
  for _ in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

  

  # My Code Starts Here 
  
  twos_test=tf.equal(tf.argmax(y,1),2) # check where I predicted 2
  
  twos_label=tf.equal(tf.argmax(y_,1),2) # check where true label is 2
  correct_prediction_of_two=tf.logical_and(twos_test, twos_label) # find where we match
  
  correct_prediction_of_two_where=tf.where(correct_prediction_of_two) # retrieve the indices of match
  
  indices_for_correctly_labelled_two=tf.to_int32(correct_prediction_of_two_where[:10]) # retrieve the first 10 indices
  
  my_indices=sess.run(indices_for_correctly_labelled_two,feed_dict={x:mnist.test.images,y_:mnist.test.labels}) # run the graph to calculate indices
  
  
  
  

  
  dydx = tf.gradients(y[0][6],x) # calculate the derivative of evidence for the label 6 w.r.t input pixels
  
  # reshaping array so that tensorflow is happy
  this_x = np.reshape(mnist.test.images[my_indices[0][0]],(1, 784)) 
  this_y=np.reshape(mnist.test.labels[my_indices[0][0]],(1,10))
  feed_dict = {x: this_x, y_: this_y}
  
  
  dyx_local=sess.run(dydx,feed_dict) # run the graph to get the derivatives
  dyx_local_signs=np.sign(dyx_local) # find the signs of derivatives

  
  noise_image=np.zeros(784) 
  noise_image=noise_image+0.25*dyx_local_signs # create the well-crafted noise
  plot_image(noise_image)
  
  
  # go over indices and add the noise to each image and trick the classifier
  for i in range(0,10):

  	plot_image(mnist.test.images[my_indices[i][0]])
  	
  	new_image=mnist.test.images[my_indices[i][0]]+noise_image
    
  	
  	plot_image(new_image)


  	
    # This part can be used to test to see how classifier fails
  	'''
    this_x = np.reshape(mnist.test.images[my_indices[i][0]],(1, 784))
    this_y=np.reshape(mnist.test.labels[my_indices[i][0]],(1,10))
  	predicted_label=tf.argmax(y,1)
  	
  	print(sess.run(predicted_label,feed_dict={x:this_x,y_:this_y}))
  	
  	print(sess.run(predicted_label,feed_dict={x:new_test_image,y_:this_y}))
  	'''

  


# this method does the plotting of images
def plot_image(data_array):
	image =  data_array.reshape([28,28])
	plt.imshow(image, cmap=plt.get_cmap('gray_r'))
	plt.show()


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--data_dir', type=str, default='/tmp/tensorflow/mnist/input_data',
                      help='Directory for storing input data')
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)