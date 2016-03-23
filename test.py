""" A demonstration of how to use the library. Using the small iris dataset.
"""


from sklearn import datasets
from stacked_autoencoder import StackedAutoencoder

iris = datasets.load_iris()
x = iris.data

sae = StackedAutoencoder(x, dims=[6, 4], depth=2)
result = sae.encode()