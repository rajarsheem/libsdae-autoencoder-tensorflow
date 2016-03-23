from basic_autoencoder import BasicAutoEncoder


class StackedAutoencoder:
    """A deep autoencoder"""

    def __init__(self, x, dims, depth=2, epoch=1000):
        self.epoch = epoch
        self.dims = dims
        self.x = x
        self.depth = depth

    def encode(self):
        data = self.x
        for i in range(self.depth):
            ae = BasicAutoEncoder(data_x=data, hidden_dim=self.dims[i], epoch=self.epoch)
            ae.run()
            data = ae.get_hidden_feature()
        return data

