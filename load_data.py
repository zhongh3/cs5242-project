import numpy as np
import pandas as pd


def load_train_data(in_height, in_width, num_rows, train_ratio):
    # To load train data and split it for training and testing
    # x: input;     y: label
    x_train, y_train, x_eval, y_eval = None, None, None, None

    print("Start loading data...")
    ###############################################
    # actual data for training and testing
    inputs = read_data('train.csv', in_height, in_width, nrows=num_rows)
    labels = read_label('train_label.csv', nrows=num_rows)
    ###############################################

    # Partition data for training and evaluation
    np.random.seed(0)
    mask = np.random.rand(inputs.shape[0]) <= train_ratio

    x_train = inputs[mask]
    y_train = labels[mask]
    x_eval = inputs[~mask]
    y_eval = labels[~mask]

    return x_train, y_train, x_eval, y_eval


def load_test_data(in_height, in_width, num_rows):
    # To load test data for prediction
    test_data = None

    test_data = read_data('test.csv', in_height, in_width, nrows=num_rows)

    return test_data


def read_data(file_name, in_height, in_width, nrows):
    directory = './'
    path = directory + file_name

    df = pd.read_csv(path, header=None, names=list(range(in_height * in_width)), nrows=nrows)  # DataFrame

    # inputs = np.nan_to_num(np.asarray(df))
    inputs = df.fillna(0).as_matrix()
    print(path, " - data shape = ", inputs.shape)

    return inputs  # numpy array: (n, (in_height x in_width))


def read_label(file_name, nrows):
    directory = './'
    path = directory + file_name

    df = pd.read_csv(path, header=0, usecols=[1], nrows=nrows)  # DataFrame

    # labels = np.nan_to_num(np.asarray(df).reshape(-1))
    labels = df.fillna(0).as_matrix().reshape(-1)

    print(path, " - label shape = ", labels.shape)

    return labels  # numpy array: (n,)
