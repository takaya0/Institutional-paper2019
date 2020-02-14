import numpy as np
from tqdm import tqdm
from copy import copy
from scipy.special import softmax
from sklearn import datasets
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt

ITER_NUM = 300
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
        self.train_Log = []
        self.Val_Log = []
        self.alpha = 0.03

    def softmax(self, Z):
        """
        Z = Z - np.max(Z)
        SUM = np.sum(np.exp(Z), axis=0)
        res = np.exp(Z)/(SUM)
        """
        return softmax(Z, axis=0)

    def train(self, train_x, train_y, test_x=[], test_y=[]):
        N = len(train_x)
        one_vec = np.ones((N, 1))
        X = copy(train_x.T)
        Y = train_y.T
        self.W = np.random.randint(-10, 10,
                                   (Clasfication_number, len(train_x[0])))
        self.b = np.random.randint(-50, 50, (Clasfication_number, 1))
        for z in range(ITER_NUM):
            Z = np.dot(self.W, X) + self.b
            F = self.softmax(Z)
            self.W = self.W - self.alpha * np.dot(F - Y, X.T)
            self.b = self.b - self.alpha * np.dot(F - Y, one_vec)
            """
            for k in range(N):
                acc_Z = np.dot(self.W, train_x[k].T).T + self.b.T
                acc_Z = self.softmax(acc_Z.T)
                print("{}番目のデータの予測値は{}, 本当は{}".format(
                    k + 1, np.argmax(acc_Z), np.argmax(train_y[k])))
            """
            acc_train = self.score(train_x, train_y)
            self.train_Log.append(acc_train)
            if z % 10 == 0:
                print('acc_score is {}'.format(acc_train))
            if len(test_x) != 0 and len(test_y) != 0:
                self.Val_Log.append(self.score(test_x, test_y))

    def predict(self, X):
        Z = np.dot(self.W, X.T).T + self.b.T
        predicted_class = self.softmax(Z.T)
        return np.argmax(predicted_class)

    def score(self, test_x, test_y):
        N = len(test_x)
        acc = 0
        for k in range(N):
            # print(self.predict(test_x[k].T))
            if self.predict(test_x[k].T) == np.argmax(test_y[k]):
                acc = acc+1
        return acc/N

    def get_Log(self, log_type='train'):
        if log_type == 'train':
            return np.array(self.train_Log)
        elif log_type == 'test':
            return np.array(self.Val_Log)
        else:
            raise 'No such a log file type {}'.format(log_type)


def main():
    Used_dataset = 'mnist_784'
    MNIST = datasets.fetch_openml(Used_dataset)
    X = MNIST.data/255.0
    Y = MNIST.target
    """
    X = np.array([[1, 2, 10, 4], [3, 4, 2, 14], [
                 2, 5, 6, 7], [32, 5, 14, 17], [11, 23, 16, 5], [18, 31, 7, 6]])
    Y = np.array([0, 2, 1, 0, 1, 2])
    for i in range(3):
        Photo = X[i + 32 * i]
        plt.imshow(Photo.reshape(28, 28), cmap='gist_gray')
        plt.savefig("MNIST_sample{}.png".format(i + 1))
    """
    Y = one_hot_vectorize(Y, Clasfication_number)
    x_train, x_test, t_train, t_test = train_test_split(
        X, Y, test_size=0.3, random_state=0)
    model = Logstic_Regression()
    model.train(x_train, t_train, test_x=x_test, test_y=t_test)
    train_log_data = model.get_Log(log_type='train')
    val_log_data = model.get_Log(log_type='test')
    plt.plot(train_log_data, label='train_loss', color='red')
    plt.plot(val_log_data, label='test_loss')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
