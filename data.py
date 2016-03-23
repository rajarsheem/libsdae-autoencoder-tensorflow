from sklearn import datasets
import numpy as np

iris = datasets.load_iris()
x = iris.data
y = iris.target


def get_batch(X, size):
    a = np.random.choice(len(X), size, replace=False)
    return X[a]


def get_full():
    return 0

