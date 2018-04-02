import tensorflow as tf
from model import model_fn
from setting import batch_size
from setting import num_steps
from load_data import load_data


def main():
    x_train, y_train, x_test, y_test = load_data()
    model = tf.estimator.Estimator(model_fn)
    # Define the input function for training
    input_fn = tf.estimator.inputs.numpy_input_fn(
        x= {'file': x_train}, y=y_train,
        batch_size=batch_size, num_epochs=None, shuffle=True)
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


if __name__ == '__main__':
    main()

