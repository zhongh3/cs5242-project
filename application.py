from load_data import load_data
from model import to_model
from train import train

from evaluation import evaluation


def main():
    x_train, y_train, x_test, y_test = load_data()
    model = to_model(x_train)
    train(model, x_train, y_train, x_test, y_test)
    evaluation(model, x_test, y_test)


if __name__ == '__main__':
    main()

