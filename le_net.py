import tensorflow as tf
import tensorflow.contrib.slim as slim

tf.logging.set_verbosity("INFO")


def le_net(x_dict, in_height, in_width, n_classes, dropout, reuse, is_training):
    # Define a scope for reusing the variables
    with tf.variable_scope('LetNet', reuse=reuse):
        # TF Estimator input is a dict, in case of multiple inputs
        x = x_dict['file']

        # Input is a vector of 4096 features (64 x 64 pixels)
        # Reshape to match picture format [Height x Width x Channel] (single channel)
        # Tensor input become 4-D: [Batch Size, Height, Width, Channel]
        x = tf.cast(tf.reshape(x, shape=[-1, in_height, in_width, 1]), tf.float32)

        # Define the structure of LeNet
        net = slim.conv2d(x, 6, [5, 5], 1, padding='SAME', scope='conv1',
                          weights_initializer=tf.truncated_normal_initializer(stddev=0.01),
                          weights_regularizer=slim.l2_regularizer(0.0005))

        net = slim.max_pool2d(net, [2, 2], scope='pool2')

        net = slim.conv2d(net, 16, [5, 5], 1, scope='conv3',
                          weights_initializer=tf.truncated_normal_initializer(stddev=0.01),
                          weights_regularizer=slim.l2_regularizer(0.0005))

        net = slim.max_pool2d(net, [2, 2], scope='pool4')

        net = slim.conv2d(net, 120, [5, 5], 1, scope='conv5',
                          weights_initializer=tf.truncated_normal_initializer(stddev=0.01),
                          weights_regularizer=slim.l2_regularizer(0.0005))

        net = slim.flatten(net, scope='flat6')

        net = slim.fully_connected(net, 84, scope='fc7')

        net = slim.dropout(net, dropout, is_training=is_training, scope='dropout8')

        out = slim.fully_connected(net, n_classes, scope='fc9')

    return out
