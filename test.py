""" A demonstration of how to use the library. Using the small iris dataset.
"""

from sklearn import datasets
import numpy as np

from deepautoencoder import StackedAutoEncoder

iris = datasets.load_iris()
x = iris.data
model = StackedAutoEncoder(dims=[5, 6], activations=['sigmoid', 'tanh'], noise='gaussian', epoch=200,
                          loss='rmse', lr=0.005)
result = model.fit_transform(x)
# model.fit(x)
# result = model.transform(np.random.rand(5, x.shape[1]))
print(result.shape)

