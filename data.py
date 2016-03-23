from sklearn import datasets
import numpy as np

iris = datasets.load_iris()
X = iris.data
Y = iris.target


def get_batch(size):
    a = np.random.choice(len(Y), size, replace=False)
    return X[a]


def get_full():
    return  X

