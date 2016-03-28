""" A demonstration of how to use the library. Using the small iris dataset.
"""

from sklearn import datasets

from deepautoencoder import StackedAutoEncoder

iris = datasets.load_iris()
x = iris.data
result = StackedAutoEncoder(x, dims=[5], activations=['sigmoid'], noise='gaussian', epoch=1000,
                            loss='cross-entropy').encode()
print(result)
print(result.shape)
