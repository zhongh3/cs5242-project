import tensorflow as tf
from model import model_fn
from setting import batch_size
from setting import num_steps
from load_data import load_data
import numpy as np

# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
# mnist = input_data.read_data_sets("/tmp/data/", one_hot=False)

def main():
    x_train, y_train, x_test, y_test = load_data()
    model = tf.estimator.Estimator(model_fn)
    x_train_cast = np.nan_to_num(x_train.reshape(500, 4096))
    y_train_cast = np.nan_to_num(y_train.reshape(500, ))

    x_test_cast = np.nan_to_num(x_test.reshape(500, 4096))
    y_test_cast = np.nan_to_num(y_test.reshape(500, ))
    print("hehehehe")
    print(y_train_cast.shape)
    print(x_train_cast.shape)

    # print("hahahahha")
    # print(mnist.train.images.shape)
    # print(mnist.train.labels.shape)
    # print(mnist.test.images.shape)
    # print(mnist.test.labels.shape)

    # Define the input function for training
    input_fn = tf.estimator.inputs.numpy_input_fn(
        x= {'file': x_train_cast}, y=y_train_cast,
        batch_size=batch_size, num_epochs=5, shuffle=True)
    # Train the Model
    model.train(input_fn, steps=num_steps)

    # Evaluate the Model
    # Define the input function for evaluating
    input_fn = tf.estimator.inputs.numpy_input_fn(
        x={'file': x_test_cast}, y=y_test_cast,
        batch_size=batch_size, shuffle=False)
    # Use the Estimator 'evaluate' method
    e = model.evaluate(input_fn)
    print("Testing Accuracy:", e['accuracy'])


if __name__ == '__main__':
    main()

