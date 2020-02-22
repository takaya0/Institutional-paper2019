import numpy as np
from matplotlib import pyplot as plt


class activation():
    def Sigmoid(self, x):
        return 1/(1 + np.exp(-x))

    def ReLU(self, x):
        return np.max(x, 0)

    def leaky_ReLU(self, a, x):
        return np.max(a * x, 0)

    def identity(self, x):
        return x


class Neural_Network(activation):
    def __init__(self, alpha=0.001):
        self.layer = []
        self.actovation = []
        self.alpha = alpha

    def add_layer(self, shapes, activation):
        self.layer.append(np.random.randint(-50, 50, shapes)
        self.activation.append(activation)
