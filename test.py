""" A demonstration of how to use the library. Using the small iris dataset.
"""

from sklearn import datasets
import numpy as np
from deepautoencoder import StackedAutoEncoder

iris = datasets.load_iris()
x = iris.data
model = StackedAutoEncoder(dims=[5], activations=['sigmoid'], noise='gaussian', epoch=500,
                           loss='rmse', lr=0.005)
model.fit(x)
result = model.transform(np.random.rand(5, x.shape[1]))
# print(result)
print(result[0])
