import tensorflow as tf
import tensorflow.contrib.slim as slim

tf.logging.set_verbosity("INFO")


def conv_net(x_dict, in_height, in_width, n_classes, dropout, reuse, is_training):
    # Define a scope for reusing the variables
    with tf.variable_scope('ConvNet', reuse=reuse):
        # TF Estimator input is a dict, in case of multiple inputs
        x = x_dict['file']

        # Input is a vector of 4096 features (64 x 64 pixels)
        # Reshape to match picture format [Height x Width x Channel] (single channel)
        # Tensor input become 4-D: [Batch Size, Height, Width, Channel]
        x = tf.cast(tf.reshape(x, shape=[-1, in_height, in_width, 1]), tf.float32)

        # Define the structure of ConvNet
        # Convolution Layer with 32 filters and a kernel size of 5x5
        conv1 = tf.layers.conv2d(x, 32, [5, 5], activation=tf.nn.relu,
                                 kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
                                 kernel_regularizer=slim.l2_regularizer(0.0005))

        # Max Pooling (down-sampling) with strides of 2 and kernel size of 2x2
        conv1 = tf.layers.max_pooling2d(conv1, [2, 2], 2)

        # Convolution Layer with 64 filters and a kernel size of 3x3
        conv2 = tf.layers.conv2d(conv1, 64, [3, 3], activation=tf.nn.relu,
                                 kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
                                 kernel_regularizer=slim.l2_regularizer(0.0005))

        # Max Pooling (down-sampling) with strides of 2 and kernel size of 2x2
        conv2 = tf.layers.max_pooling2d(conv2, [2, 2], 2)

        # Flatten the data to a 1-D vector for the fully connected layer
        fc1 = tf.contrib.layers.flatten(conv2)

        # Fully connected layer
        fc1 = tf.layers.dense(fc1, 1024)

        # Apply Dropout (if is_training is False, dropout is not applied)
        fc1 = tf.layers.dropout(fc1, rate=dropout, training=is_training)

        # Output layer, class prediction
        out = tf.layers.dense(fc1, n_classes)

    return out
