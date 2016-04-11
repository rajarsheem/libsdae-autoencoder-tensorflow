from sklearn import datasets
import numpy as np

iris = datasets.load_iris()
x = iris.data
y = iris.target


def get_batch(X, X_, size):
    a = np.random.choice(len(X), size, replace=False)
    return X[a], X_[a]


def get_full():
    return 0

