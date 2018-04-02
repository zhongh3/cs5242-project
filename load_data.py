# import tensorflow as tf
import numpy as np
import pandas as pd
# import os


def load_data():
    x_train, y_train, x_test, y_test = None, None, None, None

    # sample length to read = in_height x in_width, the rest will be truncated
    # 64 x 64 = 4096
    in_height = 64
    in_width = 64

    x_train = read_data('sample_data.csv', in_height, in_width)
    y_train = read_label('sample_label.csv')
    x_test = read_data('sample_data.csv', in_height, in_width)
    y_test = read_label('sample_label.csv')

    return x_train, y_train, x_test, y_test


def read_data(file_name, in_height, in_width):
    directory = './data/'
    path = directory + file_name
    print(path)

    # df = pd.read_csv(path, header=None)
    df = pd.read_csv(path, header=None, usecols=list(range(in_height * in_width)))  # DataFrame
    print(df.shape)

    inputs = np.asarray(df)

    inputs_3d = inputs.reshape((inputs.shape[0], in_height, in_width))
    print(inputs_3d.shape)

    return inputs_3d  # numpy array: n x in_height x in_wdith


def read_label(file_name):
    directory = './data/'
    path = directory + file_name
    print(path)

    df = pd.read_csv(path, header=0, usecols=[1])  # DataFrame

    labels = np.asarray(df)
    print(labels.shape)

    return labels  # numpy array: n x 1


# def main():
#     load_data()

