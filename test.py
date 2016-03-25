""" A demonstration of how to use the library. Using the small iris dataset.
"""

from sklearn import datasets
from stacked_autoencoder import StackedAutoencoder

iris = datasets.load_iris()
x = iris.data

result = StackedAutoencoder(
    x, dims=[5, 4, 3], noise='gaussian', epoch=1000
    ).encode()
print(result)
print(result.shape)
