import tensorflow as tf
import pandas as pd
import numpy as np
import csv

from model_fn import model_fn
from setting import batch_size, num_steps, train_epoch
from setting import in_height, in_width, num_rows, train_ratio, max_ckpt, ckpt_steps
from load_data import load_train_data, load_test_data


tf.logging.set_verbosity("INFO")


def main():
    # x: input;     y: label
    x_train, y_train, x_test, y_test = load_train_data(in_height, in_width, num_rows, train_ratio)

    model = tf.estimator.Estimator(model_fn)
    rc = tf.estimator.RunConfig(model_dir="./model", keep_checkpoint_max=max_ckpt, save_checkpoints_steps=ckpt_steps)
    model = tf.estimator.Estimator(model_fn, config=rc)

    x_predict=load_test_data(in_height, in_width, num_rows)

    # Define the input function for training
    input_fn = tf.estimator.inputs.numpy_input_fn(
        x={'file': x_train}, y=y_train,
        batch_size=batch_size, num_epochs=train_epoch, shuffle=True)
    # Train the Model
    model.train(input_fn, steps=num_steps)

    # Evaluate the Model
    # Define the input function for evaluating
    input_fn = tf.estimator.inputs.numpy_input_fn(
        x={'file': x_test}, y=y_test,
        batch_size=batch_size, shuffle=False)
    # Use the Estimator 'evaluate' method
    e = model.evaluate(input_fn)
    total_steps = e['global_step']
    print("global_step:", total_steps)
    print('Testing Accuracy = ', e['accuracy'], "Loss = ", e['loss'])

    # Evaluate Checkpoints
    total_ckpts = total_steps // ckpt_steps
    eval_results = np.zeros((total_ckpts, 3))

    for i in range(0, total_ckpts):
        j = np.min([(i + 1) * ckpt_steps + 1, total_steps])
        ckpt_path = './model/model.ckpt-' + str(j)
        print(ckpt_path)
        e = model.evaluate(input_fn, checkpoint_path=ckpt_path)
        eval_results[i, :] = [j, e['accuracy'], e['loss']]

    df = pd.DataFrame(eval_results)
    header = ["step", "accuracy", "loss"]
    df.to_csv('./eval_ckpts.csv', header=header, index=None)

    # Predict
    predict_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={'file': x_predict},
        batch_size=batch_size, num_epochs=1, shuffle=False)

    results = model.predict(input_fn=predict_input_fn)
    i = 0

    with open('result.csv', 'w') as csvfile:
        csv_writer = csv.writer(csvfile,)
        csv_writer.writerow(["sample_id", "malware"])
        for result in results:
            csv_writer.writerow([i, result[1]])
            i = i+1


if __name__ == '__main__':
    main()

