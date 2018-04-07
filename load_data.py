import numpy as np
import pandas as pd


def load_train_data(in_height, in_width, num_rows, train_ratio):
    # To load train data and split it for training and testing
    # x: input;     y: label
    x_train, y_train, x_test, y_test = None, None, None, None


    print("Start loading data...")
    ###############################################
    # sample data for local debugging
    # inputs = read_data('sample_data.csv', in_height, in_width, nrows=None)
    # labels = read_label('sample_label.csv', nrows=None)
    ###############################################
    # actual data for training and testing
    inputs = read_data('train.csv', in_height, in_width, nrows=num_rows)
    labels = read_label('train_label.csv', nrows=num_rows)
    ###############################################

    np.random.seed(0)
    mask = np.random.rand(inputs.shape[0]) <= train_ratio

    x_train = inputs[mask]
    y_train = labels[mask]
    x_test = inputs[~mask]
    y_test = labels[~mask]

    return x_train, y_train, x_test, y_test


def load_test_data(in_height, in_width, num_rows):
    # To load test data for prediction
    test_data = None

    test_data = read_data('test.csv', in_height, in_width, nrows=num_rows)

    return test_data


def read_data(file_name, in_height, in_width, nrows):
    directory = './data/'
    path = directory + file_name
    # print(path)

    df = pd.read_csv(path, header=None, names=list(range(in_height * in_width)), nrows=nrows)  # DataFrame
    # print(df.shape)

    # inputs = np.nan_to_num(np.asarray(df))
    inputs = df.fillna(0).as_matrix()
    print(path, " - data shape = ", inputs.shape)

    return inputs  # numpy array: (n, (in_height x in_width))


def read_label(file_name, nrows):
    directory = './data/'
    path = directory + file_name
    # print(path)

    df = pd.read_csv(path, header=0, usecols=[1], nrows=nrows)  # DataFrame
    #print(df.shape)

    # labels = np.nan_to_num(np.asarray(df).reshape(-1))
    labels = df.fillna(0).as_matrix().reshape(-1)

    print(path, " - label shape = ", labels.shape)

    return labels  # numpy array: (n,)

