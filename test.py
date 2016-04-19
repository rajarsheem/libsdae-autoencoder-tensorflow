import numpy as np
from sklearn import datasets
from deepautoencoder import StackedAutoEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, precision_score

iris = datasets.load_iris().data

model2 = StackedAutoEncoder(dims=[5,6], activations=['relu', 'relu'], epoch=[1000,500],
                            loss='rmse', lr=0.007, batch_size=50, print_step=200)

pp = model2.fit_transform(iris)