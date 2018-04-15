import tensorflow as tf
import numpy as np
import pandas as pd

from setting import batch_size, num_steps, train_epoch
from setting import in_height, in_width, num_rows, train_ratio, ckpt_steps, max_ckpt
from load_data import load_train_data
from build_model import build_model

tf.logging.set_verbosity("INFO")


def train():
    # Load input data and label
    # x: input;     y: label
    x_train, y_train, x_eval, y_eval = load_train_data(in_height, in_width, num_rows, train_ratio)

    model = build_model()

    # Define the input function for training
    input_fn_t = tf.estimator.inputs.numpy_input_fn(
        x={'file': x_train},
        y=y_train,
        batch_size=batch_size,
        num_epochs=train_epoch,
        shuffle=True)

    # Train the Model
    model.train(input_fn_t, steps=num_steps)

    # Define the input function for evaluating
    input_fn_e = tf.estimator.inputs.numpy_input_fn(
        x={'file': x_eval},
        y=y_eval,
        batch_size=batch_size,
        shuffle=False)

    # Evaluate the Model
    e = model.evaluate(input_fn_e)
    total_steps = e['global_step']
    print('Evaluation Accuracy = ', e['accuracy'], "Loss = ", e['loss'], "global_step = ", total_steps)

    # Evaluate Checkpoints
    # all checkpoints have to be saved locally for evaluation, otherwise the evaluation is skipped
    # max_ckpt and ckpt_steps need to be properly tuned to enable checkpoints evaluation
    total_ckpts = total_steps // ckpt_steps
    print("Total number of checkpoints required = ", total_ckpts)

    if total_ckpts <= max_ckpt:  # all checkpoints are saved
        eval_results = np.zeros((total_ckpts, 3))

        for i in range(total_ckpts):
            j = np.min([(i + 1) * ckpt_steps + 1, total_steps])
            ckpt_path = './model/model.ckpt-' + str(j)
            print(ckpt_path)
            e = model.evaluate(input_fn_e, checkpoint_path=ckpt_path)
            eval_results[i, :] = [j, e['accuracy'], e['loss']]

        df = pd.DataFrame(eval_results)
        header = ["step", "accuracy", "loss"]
        df.to_csv('./eval_ckpts.csv', header=header, index=None)
        print("Checkpoints Evaluation is completed. The results can be found at ./eval_ckpts.csv")
    else:
        print("Checkpoints Evaluation is skipped.")


if __name__ == '__main__':
    train()
