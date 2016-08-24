from sklearn import datasets
from deepautoencoder import StackedAutoEncoder

iris = datasets.load_iris().data

model = StackedAutoEncoder(
    dims=[5, 5],
    activations=['linear', 'linear'],
    epoch=[1000, 1000],
    loss='rmse',
    lr=0.007,
    batch_size=50,
    print_step=200
)

a = model.fit_transform(iris)
print(a)
