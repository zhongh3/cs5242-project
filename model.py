from __future__ import print_function

import tensorflow as tf
from model_fn import model_fn
from setting import max_ckpt, ckpt_steps

tf.logging.set_verbosity("INFO")


def build_model():

    rc = tf.estimator.RunConfig(model_dir="./model", keep_checkpoint_max=max_ckpt,
                                save_checkpoints_steps=ckpt_steps)
    model = tf.estimator.Estimator(model_fn, config=rc)

    return model
