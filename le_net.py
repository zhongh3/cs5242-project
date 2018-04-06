# Create the neural network
import tensorflow as tf
import numpy as np

def le_net(x_dict, n_classes, dropout, reuse, is_training):
    # Define a scope for reusing the variables
    with tf.variable_scope('LetNet', reuse=reuse):
        # TF Estimator input is a dict, in case of multiple inputs
        x = x_dict['file']

        # MNIST data input is a 1-D vector of 4096 features (64*64 pixels)
        # Reshape to match picture format [Height x Width x Channel]
        # Tensor input become 4-D: [Batch Size, Height, Width, Channel]

        x = tf.cast(tf.reshape(x, shape=[-1, 64, 64, 1]), tf.float32)

        # Convolution Layer with 32 filters and a kernel size of 5
        conv1 = tf.layers.conv2d(x, 6, [5, 5], 1)
        # Max Pooling (down-sampling) with strides of 2 and kernel size of 2
        conv2 = tf.layers.max_pooling2d(conv1, 2, 2)

        # Convolution Layer with 64 filters and a kernel size of 3
        conv3 = tf.layers.conv2d(conv2, 16, [5, 5], 1)
        # Max Pooling (down-sampling) with strides of 2 and kernel size of 2
        conv4 = tf.layers.max_pooling2d(conv3, 2, 2)

        conv5 = tf.layers.conv2d(conv4, 120, [5, 5], 1)
        conv6 = tf.layers.flatten(conv5)

        fc1 = tf.layers.dense(conv6, 84)
        fc2 = tf.layers.dropout(fc1, dropout)

        # Output layer, class prediction
        out = tf.layers.dense(fc2, n_classes)

    return out