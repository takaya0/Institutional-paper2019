import numpy as np
from tqdm import tqdm
from copy import copy
from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt

ITER_NUM = 100
Clasfication_number = 10


def one_hot_vectorize(data, class_num):
    OHV = []
    for d in data:
        ohv = np.zeros(class_num)
        ohv[int(d)] = 1
        OHV.append(ohv)
    return np.array(OHV)


class Logstic_Regression():
    def __init__(self):
        self.Log = []
        self.alpha = 0.001

    def softmax(self, Z):
        Z = Z - np.max(Z, axis=0)
        SUM = np.sum(np.exp(Z), axis=0)
        res = np.exp(Z)/SUM
        return res

    def train(self, train_x, train_y):
        N = len(train_x)
        one_vec = np.ones((N, 1))
        X = copy(train_x.T)
        Y = train_y.T
        self.W = np.ones((Clasfication_number, len(train_x[0]))) + 10
        self.b = np.ones((Clasfication_number, 1)) + 30
        for _ in tqdm(range(ITER_NUM)):
            Z = np.dot(self.W, X) + self.b
            F = self.softmax(Z)
            self.W = self.W - self.alpha * np.dot(Y - F, X.T)
            self.b = self.b - self.alpha * np.dot(Y - F, one_vec)
            for k in range(1):
                acc_Z = np.dot(self.W, train_x[k].T).T + self.b.T
                acc_Z = self.softmax(acc_Z)
                print(np.argmax(acc_Z))

    def predict(self, X):
        Z = np.dot(self.W, X) + self.b
        predicted_class = self.softmax(Z)
        return predicted_class

    def score(self, test_x, test_y):
        N = len(test_x)
        acc = 0
        for i in range(N):
            if test_y[self.predict(test_x[i])] == 1:
                acc = acc+1
        return acc/N

    def get_Log(self):
        return np.array(self.Log)


def main():
    Used_dataset = 'mnist_784'
    MNIST = datasets.fetch_openml(Used_dataset)
    X = MNIST.data/255.0
    Y = MNIST.target
    """
    for i in range(3):
        Photo = X[i + 32 * i]
        plt.imshow(Photo.reshape(28, 28), cmap='gist_gray')
        plt.savefig("MNIST_sample{}.png".format(i + 1))
    """
    Y = one_hot_vectorize(Y, Clasfication_number)
    x_train, x_test, t_train, t_test = train_test_split(X, Y, test_size=0.3)
    model = Logstic_Regression()
    model.train(x_train, t_train)
    log_data = model.get_Log()
    plt.plot(log_data)
    plt.show()


if __name__ == "__main__":
    main()
