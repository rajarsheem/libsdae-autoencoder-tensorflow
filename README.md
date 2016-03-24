# libsdae - Deep-Autoencoder & Denoising Autoencoder
This is a very simple library for Stacked autoencoder and denoising AE.

### What is an Autoencoder, Stacked Autoencoder, Stacked Denoising AutoEncoder ?
-> http://www.jmlr.org/papers/volume11/vincent10a/vincent10a.pdf

-> http://ufldl.stanford.edu/wiki/index.php/Stacked_Autoencoders

### Usage :
```python
x_ = StackedAutoencoder(x, dims=[5, 4, 3], noise='gaussian', epoch=1000).encode()
```

If noise is not given, it becomes an autoencoder instead of denoising autoencoder.

dims refers to the dimenstions of hidden layers. (3 layers in this case)

noise = ['gaussian', 'mask-0.4']. mask-0.4 means 40% of bits will be masked for each example.

x_ is the encoded feature representation of x.

test.py has simple usage
