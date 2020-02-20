import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


ITER_NUM = 1000


def Func(x):
    return np.sin(x) + np.power(x, 2)


def gradient_func(x):
    return np.cos(x) + 2 * x


class Gradient_Decent_Method():
    def Gradient_Decent(self, alpha=0.001):
        theta = np.random.randint(1, 100)
        Log = [theta]
        for _ in range(ITER_NUM):
            theta = theta - alpha * gradient_func(theta)
            Log.append(theta)
        self.Log = np.array(Log)


class Optimaization_Method():
    def __init__(self, EPS=0.0001):
        self.EPS = EPS

    def Partial_Derivative(self, F, point, index):
        h = np.zeros_like(point)
        h[index] = self.EPS
        return (F(point + h) - F(point - h))/(self.EPS * 2)

    def Gradient(self, F, x):
        grad = np.array([self.Partial_Derivative(F, x, i)
                         for i in range(len(x))])
        return grad

    def Gradient_Decent(self, F, alpha=0.001):
        Iteration = 3600
        theta = [9.5]
        log = [theta]
        for _ in range(Iteration):
            theta = theta - alpha * self.Gradient(Func, theta)
            log.append(theta)
        return log


def main():
    model = Optimaization_Method()
    Log = model.Gradient_Decent(Func)[0::400]
    X = np.arange(-10, 10, 0.2)
    print(Log)
    Y = []
    Log_Y = []
    for x in X:
        Y.append(Func([x]))
    for y in Log:
        Log_Y.append(Func([y]))
    Y = np.array(Y)
    Log_Y = np.array(Log_Y)
    plt.plot(X, Y)
    plt.scatter(Log, Log_Y)
    plt.savefig('Gradient_Decent.png')


if __name__ == "__main__":
    main()
