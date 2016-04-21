from sklearn import datasets

from deepautoencoder import StackedAutoEncoder

iris = datasets.load_iris().data

model2 = StackedAutoEncoder(dims=[5, 4], activations=['relu', 'relu'], epoch=[1000, 500],
                            loss='rmse', lr=0.007, batch_size=50, print_step=200)

model2.fit(iris)
