import tensorflow as tf
from model import model_fn
from setting import batch_size
from setting import num_steps
from load_data import load_train_data
from load_data import load_test_data

import csv
# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
# mnist = input_data.read_data_sets("/tmp/data/", one_hot=False)

def main():
    # x: input;     y: label
    x_train, y_train, x_test, y_test = load_train_data()
    model = tf.estimator.Estimator(model_fn)

    x_predict=load_test_data()
    # print("hahahahha")
    # print(mnist.train.images.shape)
    # print(mnist.train.labels.shape)
    # print(mnist.test.images.shape)
    # print(mnist.test.labels.shape)

    # Define the input function for training
    input_fn = tf.estimator.inputs.numpy_input_fn(
        x= {'file': x_train}, y=y_train,
        batch_size=batch_size, num_epochs=5, shuffle=True)
    # Train the Model
    model.train(input_fn, steps=num_steps)

    # Evaluate the Model
    # Define the input function for evaluating
    input_fn = tf.estimator.inputs.numpy_input_fn(
        x={'file': x_test}, y=y_test,
        batch_size=batch_size, shuffle=False)
    # Use the Estimator 'evaluate' method
    e = model.evaluate(input_fn)
    print("Testing Accuracy:", e['accuracy'])

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

