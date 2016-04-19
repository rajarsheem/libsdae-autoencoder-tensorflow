import numpy as np
from deepautoencoder import BasicAutoEncoder
import tensorflow as tf

allowed_activations = ['sigmoid', 'tanh', 'softmax', 'relu']
allowed_noises = [None, 'gaussian', 'mask']
allowed_losses = ['rmse', 'cross-entropy']


class StackedAutoEncoder:
    """A deep autoencoder"""

    def assertions(self):
        global allowed_activations, allowed_noises, allowed_losses
        assert self.loss in allowed_losses, 'Incorrect loss given'
        assert 'list' in str(type(self.dims)), 'dims must be a list even if there is one layer.'
        assert len(self.epoch) == len(self.dims), "No. of epochs must equal to no. of hidden layers"
        assert len(self.activations) == len(self.dims), "No. of activations must equal to no. of hidden layers"
        assert all(True if x > 0 else False for x in self.epoch), "No. of epoch must be atleast 1"
        assert set(self.activations + allowed_activations) == set(allowed_activations), "Incorrect activation given."
        assert self.noise in allowed_noises, "Incorrect noise given"

    def __init__(self, dims, activations, epoch=1000, noise=None, loss='rmse', lr=0.001, batch_size=100, print_step=50):
        self.print_step = print_step
        self.batch_size = batch_size
        self.lr = lr
        self.ae = None
        self.loss = loss
        self.activations = activations
        self.noise = noise
        self.epoch = epoch
        self.dims = dims
        self.assertions()
        self.depth = len(dims)
        self.weights, self.biases = [], []

    def add_noise(self, x):
        if self.noise == 'gaussian':
            n = np.random.normal(0, 0.1, (len(x), len(x[0])))
            return x + n
        if 'mask' in self.noise:
            frac = float(self.noise.split('-')[1])
            temp = np.copy(x)
            for i in temp:
                n = np.random.choice(len(i), round(frac * len(i)), replace=False)
                i[n] = 0
            return temp
        if self.noise == 'sp':
            pass

    def fit(self, x):
        for i in range(self.depth):
            print('Layer {0}'.format(i + 1))
            if self.noise is None:
                self.ae = BasicAutoEncoder(data_x=x, activation=self.activations[i], data_x_=x,
                                           hidden_dim=self.dims[i], epoch=self.epoch[i], loss=self.loss,
                                           batch_size=self.batch_size, lr=self.lr, print_step=self.print_step)
            else:
                self.ae = BasicAutoEncoder(data_x=self.add_noise(x), activation=self.activations[i], data_x_=x,
                                           hidden_dim=self.dims[i],
                                           epoch=self.epoch[i], loss=self.loss, batch_size=self.batch_size, lr=self.lr,
                                           print_step=self.print_step)
            x, w, b = self.ae.run()
            self.weights.append(w)
            self.biases.append(b)

    def transform(self, data):
        sess = tf.Session()
        x = tf.constant(data, dtype=tf.float32)
        for w, b, a in zip(self.weights, self.biases, self.activations):
            weight = tf.constant(w, dtype=tf.float32)
            bias = tf.constant(b, dtype=tf.float32)
            layer = tf.matmul(x, weight) + bias
            x = self.ae.activate(layer, a)
        return x.eval(session=sess)

    def fit_transform(self, x):
        self.fit(x)
        return self.transform(x)
