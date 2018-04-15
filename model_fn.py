from __future__ import print_function

import tensorflow as tf
from conv_net import conv_net
from le_net import le_net
from setting import in_height, in_width, num_classes, dropout, learning_rate

tf.logging.set_verbosity("INFO")


def model_fn(features, labels, mode):
    # Build the neural network

    # dropout should only be activated during training (train), not evaluation or prediction (test)
    # need to create 2 logits to share the same weights

    ##################################################################
    # LeNet: activated by default
    logits_train = le_net(features, in_height, in_width, num_classes, dropout, reuse=False, is_training=True)
    logits_test = le_net(features, in_height, in_width, num_classes, dropout, reuse=True, is_training=False)

    # ConvNet: uncomment the two rows below to activate
    # logits_train = conv_net(features, in_height, in_width, num_classes, dropout, reuse=False, is_training=True)
    # logits_test = conv_net(features, in_height, in_width, num_classes, dropout, reuse=True, is_training=False)
    ##################################################################

    # Predictions
    pred_classes = tf.argmax(logits_test, axis=1)   # output is class label
    pred_probas = tf.nn.softmax(logits_test)        # output is probabilities of being each class

    # Output probabilities for prediction
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode, predictions=pred_probas)

    # Define loss and optimizer
    loss_op = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=logits_train,
        labels=tf.cast(labels, dtype=tf.int32)))

    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

    train_op = optimizer.minimize(loss_op,
                                  global_step=tf.train.get_global_step())

    # Evaluate the accuracy of the model
    acc_op = tf.metrics.accuracy(labels=labels, predictions=pred_classes)

    estim_specs = tf.estimator.EstimatorSpec(
        mode=mode,
        predictions=pred_classes,
        loss=loss_op,
        train_op=train_op,
        eval_metric_ops={'accuracy': acc_op})

    return estim_specs
