[![Build Status](https://travis-ci.org/rajarsheem/libsdae.svg?branch=master)](https://travis-ci.org/rajarsheem/libsdae)
# Deep-Autoencoder & Denoising Autoencoder
Requirements: Tensorflow and numpy.

### What is an Autoencoder, Stacked Autoencoder, Stacked Denoising AutoEncoder ?
-> http://www.jmlr.org/papers/volume11/vincent10a/vincent10a.pdf

-> http://ufldl.stanford.edu/wiki/index.php/Stacked_Autoencoders

### Install:
```
git clone https://github.com/rajarsheem/libsdae.git
python3 setup.py
```

### Usage :
```python
from deepautoencoder import StackedAutoEncoder
model = StackedAutoEncoder(dims=[5], activations=['sigmoid'], noise='gaussian', epoch=500,
                           loss='rmse')
# encoding same data                           
result = model.fit_transform(x)
# fitting on one dataset and encoding a random data
model.fit(x)
result = model.transform(np.random.rand(5, x.shape[1]))
```

If noise is not given, it becomes an autoencoder instead of denoising autoencoder.

dims refers to the dimenstions of hidden layers. (3 layers in this case)

noise = (optional)['gaussian', 'mask-0.4']. mask-0.4 means 40% of bits will be masked for each example.

x_ is the encoded feature representation of x.

loss = (optional) reconstruction error. rmse or softmax with cross entropy are allowed. default is rmse.

test.py has simple usage

Note: If you find any issue or scope of improvements, please be kind and report to rajarsheem@gmail.com or raise an issue here.

Regards.
