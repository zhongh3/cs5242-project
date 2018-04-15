import tensorflow as tf
import numpy as np
import pandas as pd
# import csv

from setting import batch_size, in_height, in_width, num_rows
from load_data import load_test_data
from build_model import build_model

tf.logging.set_verbosity("INFO")


def predict(model):
    # Load data to carry out prediction
    # x: input;     y: label
    x_predict = load_test_data(in_height, in_width, num_rows)

    # Define the input function for prediction
    predict_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={'file': x_predict},
        batch_size=batch_size,
        num_epochs=1,
        shuffle=False)

    # Use the model for prediction
    pred_results = model.predict(input_fn=predict_input_fn)

    # results[:, 0] is the probability of class 0 (i.e. not malware)
    # results[:, 1] is the probability of class 1 (i.e. being malware)
    results = np.asarray(list(pred_results))
    df_out = pd.DataFrame(results[:, 1])

    header = ["malware"]
    df_out.to_csv('./result.csv', header=header, index=True, index_label="sample_id")

    # i = 0
    # with open('result.csv', 'w') as csvfile:
    #     csv_writer = csv.writer(csvfile,)
    #     csv_writer.writerow(["sample_id", "malware"])
    #     for result in pred_results:
    #         csv_writer.writerow([i, result[1]])
    #         i = i+1

    print('You can find the prediction results in ./result.csv.')


if __name__ == '__main__':
    predict(build_model())
