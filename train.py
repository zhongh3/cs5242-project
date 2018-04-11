import tensorflow as tf

from model import model_fn
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

    x_predict = load_test_data(in_height, in_width, num_rows)

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

if __name__ == '__main__':
    main()
