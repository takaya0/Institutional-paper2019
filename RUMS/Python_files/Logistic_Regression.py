import numpy as np
from tqdm import tqdm
from copy import copy
from scipy.special import softmax
from sklearn import datasets
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
import pandas as pd
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-e', '--Epoch', default=500, type=int)
parser.add_argument('-b', '--Batch', default=50, type=int)
args = parser.parse_args()
Classification_number = 10

Epoch = args.Epoch
bacth = args.Batch


def one_hot_vectorize(data, class_num):
    OHV = []
    for d in data:
        ohv = np.zeros(class_num)
        ohv[int(d)] = 1
        OHV.append(ohv)
    return np.array(OHV)


class Logistic_Regression():
    def __init__(self):
        self.train_Log = []
        self.Val_Log = []
        self.alpha = 0.0001

    def softmax(self, X):
        return softmax(X, axis=0)

    def train(self, train_x, train_y, test_x=[], test_y=[]):
        N = len(train_x)
        one_vec = np.ones((N, 1))
        X = copy(train_x.T)
        Y = train_y.T
        self.W = np.random.randint(-30, 30,
                                   (Classification_number, len(train_x[0])))
        self.b = np.random.randint(-50, 50, (Classification_number, 1))
        for _ in tqdm(range(Epoch)):
            Z = np.dot(self.W, X) + self.b
            F = self.softmax(Z)
            self.W = self.W - self.alpha * np.dot(F - Y, X.T)
            self.b = self.b - self.alpha * np.dot(F - Y, one_vec)

            acc_train = self.score(train_x, train_y)
            self.train_Log.append(acc_train)
            if len(test_x) != 0 and len(test_y) != 0:
                self.Val_Log.append(self.score(test_x, test_y))

    def SGD_train(self, train_x, train_y, test_x=[], test_y=[]):
        N = len(train_x)
        index = np.arange(0, N, 1)
        self.W = np.random.randint(-30, 30,
                                   (Classification_number, len(train_x[0])))
        self.b = np.random.randint(-50, 50, (Classification_number, 1))
        pbar = tqdm(range(Epoch))
        train_score = 0
        for _ in pbar:
            pbar.set_description("train_score = {}".format(train_score))
            X = train_x
            Y = train_y
            np.random.shuffle(index)
            splited_index = np.array_split(index, bacth)
            for ith_batch in splited_index:
                batch_X = X[ith_batch]
                batch_Y = Y[ith_batch]
                one_vec = np.ones((len(batch_X), 1))
                batch_X = copy(batch_X.T)
                batch_Y = copy(batch_Y.T)
                Z = np.dot(self.W, batch_X) + self.b
                F = self.softmax(Z)
                self.W = self.W - self.alpha * np.dot(F - batch_Y, batch_X.T)
                self.b = self.b - self.alpha * np.dot(F - batch_Y, one_vec)
            acc_train = self.score(train_x, train_y)
            train_score = acc_train
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
    for i in range(3):
        Photo = X[i + 32 * i]
        plt.imshow(Photo.reshape(28, 28), cmap='gist_gray')
        plt.savefig("../Images/MNIST_sample{}.eps".format(i + 1))
    """
    Y = one_hot_vectorize(Y, Classification_number)
    x_train, x_test, t_train, t_test = train_test_split(
        X, Y, test_size=0.3, random_state=0)
    model = Logistic_Regression()
    model.train(x_train, t_train, test_x=x_test, test_y=t_test)
    train_log_data = model.get_Log(log_type='train')
    val_log_data = model.get_Log(log_type='test')
    plt.plot(train_log_data, label='train_score',
             color='red', linestyle='dashed')
    plt.plot(val_log_data, label='test_score')
    plt.xlabel('Epoch number')
    plt.ylabel('Accuracy score')
    plt.title("Logistic Regression with GD")
    plt.legend()
    # plt.show()
    plt.savefig('../Images/MNIST_Experiment.eps')
    """


if __name__ == "__main__":
    main()
