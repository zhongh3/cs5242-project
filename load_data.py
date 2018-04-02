import numpy as np
import pandas as pd


def load_data():
    x_train, y_train, x_test, y_test = None, None, None, None

    # sample length to read = in_height x in_width, the rest will be truncated
    # 64 x 64 = 4096
    in_height = 64
    in_width = 64

    ###############################################
    # sample data for local debugging
    x_train = read_data('sample_data.csv', in_height, in_width)
    y_train = read_label('sample_label.csv')
    x_test = read_data('sample_data.csv', in_height, in_width)
    y_test = np.zeros(y_train.shape[0])
    ###############################################
    # actual data for training and testing
    # x_train = read_data('train.csv', in_height, in_width)
    # y_train = read_label('train_label.csv')
    # x_test = read_data('test.csv', in_height, in_width)
    # y_test = np.zeros(y_train.shape[0])
    ###############################################

    return x_train, y_train, x_test, y_test


def read_data(file_name, in_height, in_width):
    directory = './data/'
    path = directory + file_name
    # print(path)

    df = pd.read_csv(path, header=None, names=list(range(in_height * in_width)))  # DataFrame
    print(df.shape)

    inputs = np.nan_to_num(np.asarray(df))
    print(path, " - data shape = ", inputs.shape)

    return inputs  # numpy array: (n, (in_height x in_wdith))


def read_label(file_name):
    directory = './data/'
    path = directory + file_name
    # print(path)

    df = pd.read_csv(path, header=0, usecols=[1])  # DataFrame
    print(df.shape)

    labels = np.nan_to_num(np.asarray(df).reshape(-1))
    print(path, " - label shape = ", labels.shape)

    return labels  # numpy array: (n,)

