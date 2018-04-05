import numpy as np
import pandas as pd

from setting import num_rows
from setting import train_ratio

def load_train_data():
    # To load train data and split it for training and testing
    # x: input;     y: label
    x_train, y_train, x_test, y_test = None, None, None, None

    # sample length to read = in_height x in_width, the rest will be truncated
    # e.g. 64 x 64 = 4096 bytes per sample
    in_height = 64
    in_width = 64

    # train_ratio = 0.8 --> split data as 80% training, 20% testing

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


def load_test_data():
    # To load test data for prediction
    test_data = None

    # sample length to read = in_height x in_width, the rest will be truncated
    # e.g. 64 x 64 = 4096 bytes per sample
    # keep the same size as in load_train_data()
    in_height = 64
    in_width = 64

    # test_data = read_data('test.csv', in_height, in_width, nrows=100)
    test_data = read_data('test.csv', in_height, in_width, nrows=num_rows)

    return test_data


def read_data(file_name, in_height, in_width, nrows):
    directory = './data/'
    path = directory + file_name
    # print(path)

    df = pd.read_csv(path, header=None, names=list(range(in_height * in_width)), nrows=nrows)  # DataFrame
    print(df.shape)

    inputs = np.nan_to_num(np.asarray(df))
    print(path, " - data shape = ", inputs.shape)

    return inputs  # numpy array: (n, (in_height x in_width))


def read_label(file_name, nrows):
    directory = './data/'
    path = directory + file_name
    # print(path)

    df = pd.read_csv(path, header=0, usecols=[1], nrows=nrows)  # DataFrame
    #print(df.shape)

    labels = np.nan_to_num(np.asarray(df).reshape(-1))
    print(path, " - label shape = ", labels.shape)

    return labels  # numpy array: (n,)

