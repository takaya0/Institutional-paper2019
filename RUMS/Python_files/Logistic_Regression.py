import numpy as np
from tqdm import tqdm
from copy import copy
from scipy.special import softmax
from sklearn import datasets
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
import pandas as pd
ITER_NUM = 30
Classification_number = 10


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
        self.alpha = 0.001

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
        self.W = np.random.randint(-30, 30,
                                   (Classification_number, len(train_x[0])))
        self.b = np.random.randint(-50, 50, (Classification_number, 1))
        for _ in tqdm(range(ITER_NUM)):
            Z = np.dot(self.W, X) + self.b
            F = self.softmax(Z)
            self.W = self.W - self.alpha * np.dot(F - Y, X.T)
            self.b = self.b - self.alpha * np.dot(F - Y, one_vec)
            acc_train = self.score(train_x, train_y)
            self.train_Log.append(acc_train)
            if len(test_x) != 0 and len(test_y) != 0:
                self.Val_Log.append(self.score(test_x, test_y))

    def predict(self, X):
        Z = np.dot(self.W, X.T).T + self.b.T
        predicted_class = self.softmax(Z.T)
        return np.argmax(predicted_class)

    def score(self, test_x, test_y):
        N = len(test_x)
        acc = 0
        """
        for k in range(N):
            # print(self.predict(test_x[k].T))
            if self.predict(test_x[k].T) == np.argmax(test_y[k]):
                acc = acc+1
        print(self.predict(test_x))
        """
        d = np.array([self.predict(test_x[k]) == np.argmax(test_y[k])
                      for k in range(N)])
        acc = np.count_nonzero(d)
        return acc/N

    def save_coef(self, file_name='coef.csv'):
        df_W = pd.DataFrame(self.W)
        df_b = pd.DataFrame(self.b.T)
        df = pd.concat([df_W, df_b])
        df.to_csv(file_name)

    def load_coef(self, file_name='coef.csv'):
        load_df = np.array(pd.read_csv(file_name, index_col=0).values.tolist())
        shape = load_df.shape
        self.W = load_df[0:shape[0] - 1]
        self.b = load_df[shape[0] - 1].T

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
    Y = one_hot_vectorize(Y, Classification_number)
    x_train, x_test, t_train, t_test = train_test_split(
        X, Y, test_size=0.3, random_state=0)
    model = Logstic_Regression()
    model.train(x_train, t_train, test_x=x_test, test_y=t_test)
    train_log_data = model.get_Log(log_type='train')
    val_log_data = model.get_Log(log_type='test')
    plt.plot(train_log_data, label='train_score',
             color='red', linestyle='dashed')
    plt.plot(val_log_data, label='test_score')
    plt.xlabel('Iteration number')
    plt.ylabel('Accuracy score')
    plt.legend()
    plt.show()
    # model.save_coef()
    # model.load_coef()
    # plt.savefig('Logistic_Regression_MNIST.png')


if __name__ == "__main__":
    main()
