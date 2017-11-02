# libsdae - deep-Autoencoder & denoising autoencoder

A simple Tensorflow based library for Deep autoencoder and denoising AE. Library follows sklearn style.

### Prerequisities & Support
* Tensorflow 1.0 is needed.
* Supports both Python 2.7 and 3.4+ . Inform if it doesn't.

## Installing
```
pip install git+https://github.com/rajarsheem/libsdae.git
```

## Usage and small doc
<i>test.ipynb</i> has small example where both a tiny and a large dataset is used.

```python
from deepautoencoder import StackedAutoEncoder
model = StackedAutoEncoder(dims=[5,6], activations=['relu', 'relu'], noise='gaussian', epoch=[10000,500],
                            loss='rmse', lr=0.007, batch_size=50, print_step=2000)
# usage 1 - encoding same data                           
result = model.fit_transform(x)
# usage 2 - fitting on one dataset and transforming (encoding) on another data
model.fit(x)
result = model.transform(np.random.rand(5, x.shape[1]))
```
![Alt text](libsdae.png?raw=true "Demo for MNIST data")

### Important points:
* If noise is not given, it becomes an autoencoder instead of denoising autoencoder.
* dims refers to the dimenstions of hidden layers. (3 layers in this case)
* noise = (optional)['gaussian', 'mask-0.4']. mask-0.4 means 40% of bits will be masked for each example.
* x_ is the encoded feature representation of x.
* loss = (optional) reconstruction error. rmse or softmax with cross entropy are allowed. default is rmse.
* print_step is the no. of steps to skip between two loss prints.
* activations can be 'sigmoid', 'softmax', 'tanh' and 'relu'.
* batch_size is the size of batch in every epoch
* Note that while running, global loss means the loss on the total dataset and not on a specific batch.
* epoch is a list denoting the no. of iterations for each layer.

### Citing

* Stacked Denoising Autoencoders: Learning Useful Representations in a Deep Network with a Local Denoising Criterion
  by P. Vincent, H. Larochelle, I. Lajoie, Y. Bengio and P. Manzagol (Journal of Machine Learning Research 11 (2010) 3371-3408)

### Contributing
You are free to contribute by starting a pull request. Some suggestions are:
* Variational Autoencoders
* Recurrent Autoencoders.
