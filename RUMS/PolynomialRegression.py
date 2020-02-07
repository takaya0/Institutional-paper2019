import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


class PolynomialRegression():
    '''
    degree is a Hyperparametor, which decide degree of 
    Polynomial of your model. (default = 3)

    '''

    def __init__(self, degree=3):
        self.degree = degree

    def h(self, x):
        res = np.poly1d(self.W[::-1])
        # print(res)
        return res(x)

    def fit(self, x_train, t_train, alpha=600):
        X = np.ones((len(x_train), self.degree + 1))
        for i in range(len(x_train)):
            for k in range(1, self.degree + 1):
                X[i][k] = np.power(x_train[i], k)
        regular = alpha * np.eye(int(self.degree + 1))
        W = np.dot(np.linalg.inv(np.dot(X.T, X) + regular), X.T)
        W = np.dot(W, t_train)
        self.W = W

    def score(self, x_test, t_test):
        accuracy = 0
        for i in range(len(x_test)):
            accuracy += np.power(self.h(x_test[i]) - t_test[i], 2)
        accuracy = accuracy * (1/len(x_test))
        return 100 - np.sqrt(accuracy)

    def plot_result(self, X, Y):
        plt.scatter(X, Y)
        N = np.arange(0.5, 25.5, 0.1)
        plt.plot(N, self.h(N))

    def app(self, x):
        res = self.h(x) - x**3 - 6 * x**2 - 4 * x
        return np.abs(res)


x_train = np.array([1, 3, 5, 10, 17, 25])
t_train = np.array([8, 25, 40, 55, 34, 70])
model = PolynomialRegression(degree=4)
model.fit(x_train, t_train)
result = model.score(x_train, t_train)
model.plot_result(x_train, t_train)
plt.show()
# plt.savefig('RUMS/overfitting_PR.png')
