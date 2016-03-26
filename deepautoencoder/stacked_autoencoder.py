import numpy as np
from deepautoencoder import BasicAutoEncoder


class StackedAutoEncoder:
    """A deep autoencoder"""

    def __init__(self, x, dims, epoch=1000, noise=None):
        self.noise = noise
        self.epoch = epoch
        self.dims = dims
        self.x = x
        self.depth = len(dims)

    def add_noise(self):
        if self.noise == 'gaussian':
            n = np.random.normal(0, 0.1, (len(self.x), len(self.x[0])))
            return self.x + n
        if 'mask' in self.noise:
            frac = float(self.noise.split('-')[1])
            temp = np.copy(self.x)
            for i in temp:
                n = np.random.choice(len(i), round(frac * len(i)), replace=False)
                i[n] = 0
            return temp
        if self.noise == 'sp':
            pass

    def encode(self):

        ae = None
        for i in range(self.depth):
            if self.noise is None:
                ae = BasicAutoEncoder(data_x=self.x, data_x_=self.x, hidden_dim=self.dims[i], epoch=self.epoch)
            else:
                ae = BasicAutoEncoder(data_x=self.add_noise(), data_x_=self.x, hidden_dim=self.dims[i],
                                      epoch=self.epoch)
            ae.run()
            self.x = ae.get_hidden_feature()
        return self.x
