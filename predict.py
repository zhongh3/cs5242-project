import tensorflow as tf
import csv

from model import build_model
from setting import batch_size
from setting import in_height, in_width, num_rows
from load_data import load_test_data

tf.logging.set_verbosity("INFO")


def predict(model):
    x_predict = load_test_data(in_height, in_width, num_rows)

    predict_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={'file': x_predict},
        batch_size=batch_size, num_epochs=1, shuffle=False)

    results = model.predict(input_fn=predict_input_fn)

    i = 0
    with open('result.csv', 'w') as csvfile:
        csv_writer = csv.writer(csvfile, )
        csv_writer.writerow(["sample_id", "malware"])
        for result in results:
            csv_writer.writerow([i, result[1]])
            i = i + 1

    print('You can find the prediction results in result.csv.')


if __name__ == '__main__':
    predict(build_model())


