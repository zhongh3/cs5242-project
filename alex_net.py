import tensorflow as tf
import tensorflow.contrib.slim as slim

tf.logging.set_verbosity("INFO")


def alex_net(x_dict, in_height, in_width, n_classes, dropout, reuse, is_training):
    # Define a scope for reusing the variables
    with tf.variable_scope('AlexNet', reuse=reuse):
        # TF Estimator input is a dict, in case of multiple inputs
        x = x_dict['file']

        # Reshape to match picture format [Height x Width x Channel]
        # Tensor input become 4-D: [Batch Size, Height, Width, Channel]

        x = tf.cast(tf.reshape(x, shape=[-1, in_height, in_width, 1]), tf.float32)

        net = slim.conv2d(x, 32, [5, 5], 2, padding='VALID', scope='conv1',
                          weights_initializer=tf.truncated_normal_initializer(stddev=0.01),
                          weights_regularizer=slim.l2_regularizer(0.0005))  # 64x11x11 stride = 4
        net = slim.max_pool2d(net, [2, 2], 1, scope='pool2')  # 3x3 stride = 2
        net = slim.conv2d(net, 96, [3, 3], padding='SAME', scope='conv3',  # 192x5x5
                          weights_initializer=tf.truncated_normal_initializer(stddev=0.01),
                          weights_regularizer=slim.l2_regularizer(0.0005))
        net = slim.max_pool2d(net, [3, 3], 1, scope='pool4')  # stride = 2
        net = slim.conv2d(net, 192, [3, 3], padding='SAME', scope='conv5',
                          weights_initializer=tf.truncated_normal_initializer(stddev=0.01),
                          weights_regularizer=slim.l2_regularizer(0.0005))  # 384x3x3
        net = slim.conv2d(net, 192, [3, 3], padding='SAME', scope='conv6',
                          weights_initializer=tf.truncated_normal_initializer(stddev=0.01),
                          weights_regularizer=slim.l2_regularizer(0.0005))  # 384x3x3
        net = slim.conv2d(net, 128, [3, 3], padding='SAME', scope='conv7',
                          weights_initializer=tf.truncated_normal_initializer(stddev=0.01),
                          weights_regularizer=slim.l2_regularizer(0.0005))  # 256x3x3
        net = slim.max_pool2d(net, [3, 3], 1, scope='pool8')    # stride = 2

        net = slim.conv2d(net, 2048, [2, 2], padding='SAME',
                          weights_initializer=tf.truncated_normal_initializer(stddev=0.005),
                          biases_initializer=tf.constant_initializer(0.1), scope='fc9')  # 4096x5x5
        net = slim.dropout(net, dropout, is_training=is_training, scope='dropout10')
        net = slim.conv2d(net, 1024, [1, 1], padding='SAME',
                          weights_initializer=tf.truncated_normal_initializer(stddev=0.005),
                          biases_initializer=tf.constant_initializer(0.1), scope='fc11')  # 4096x1x1
        net = slim.dropout(net, dropout, is_training=is_training, scope='dropout12')
        # out = slim.conv2d(net, 2, [1, 1], activation_fn=None,
        #                   weights_initializer=tf.truncated_normal_initializer(stddev=0.005), scope='fc13')
        net = slim.flatten(net, scope='flat13')
        out = slim.fully_connected(net, 2, scope='fc14')

    return out