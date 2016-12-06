from deepautoencoder import StackedAutoEncoder
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
data, target = mnist.train.images, mnist.train.labels

# train / test  split
idx = np.random.rand(data.shape[0]) < 0.8
train_X, train_Y = data[idx], target[idx]
test_X, test_Y = data[~idx], target[~idx]

model = StackedAutoEncoder(dims=[200, 200], activations=['linear', 'linear'], epoch=[
                           3000, 3000], loss='rmse', lr=0.007, batch_size=100, print_step=200)
model.fit(train_X)
test_X_ = model.transform(test_X)
