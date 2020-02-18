from sklearn import datasets
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC, SVC
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


# Load data from https://www.openml.org/d/554
X, y = datasets.fetch_openml('mnist_784', version=1, return_X_y=True)

X_train, X_test, y_train, y_test = train_test_split(X / 255,  # ピクセル値が 0 - 1 になるようにする
                                                    # 正解データを数値にする
                                                    y.astype('int64'),
                                                    stratify=y,
                                                    random_state=0)


def sklearn_logistic():
    clf = LogisticRegression(solver='lbfgs', multi_class='auto')
    clf.fit(X_train, y_train)  # 学習
    print('accuracy_score: %.3f' % clf.score(X_test, y_test))  # 検証


sklearn_logistic()
