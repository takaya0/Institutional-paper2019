import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt


def one_hot_vectorize(data, class_num):
    OHV = []
    for d in data:
        ohv = np.zeros(class_num)
        ohv[int(d)] = 1
        OHV.append(ohv)
    return np.array(OHV)


def main():
    Used_dataset = 'mnist_784'
    MNIST = datasets.fetch_openml(Used_dataset)
    X = MNIST.data/255.0
    Y = MNIST.target
    Y = one_hot_vectorize(Y, 10)
    #x_train, x_test, t_train, t_test = train_test_split(X, Y, test_size=0.3)


if __name__ == "__main__":
    main()
