import numpy as np
import deepautoencoder.utils as utils
import tensorflow as tf

allowed_activations = ['sigmoid', 'tanh', 'softmax', 'relu', 'linear']
allowed_noises = [None, 'gaussian', 'mask']
allowed_losses = ['rmse', 'cross-entropy']


class StackedAutoEncoder:
    """A deep autoencoder with denoising capability"""

    def assertions(self):
        global allowed_activations, allowed_noises, allowed_losses
        assert self.loss in allowed_losses, 'Incorrect loss given'
        assert 'list' in str(
            type(self.dims)), 'dims must be a list even if there is one layer.'
        assert len(self.epoch) == len(
            self.dims), "No. of epochs must equal to no. of hidden layers"
        assert len(self.activations) == len(
            self.dims), "No. of activations must equal to no. of hidden layers"
        assert all(
            True if x > 0 else False
            for x in self.epoch), "No. of epoch must be atleast 1"
        assert set(self.activations + allowed_activations) == set(
            allowed_activations), "Incorrect activation given."
        assert utils.noise_validator(
            self.noise, allowed_noises), "Incorrect noise given"

    def __init__(self, dims, activations, epoch=1000, noise=None, loss='rmse',
                 lr=0.001, batch_size=100, print_step=50):
        self.print_step = print_step
        self.batch_size = batch_size
        self.lr = lr
        self.loss = loss
        self.activations = activations
        self.noise = noise
        self.epoch = epoch
        self.dims = dims
        self.assertions()
        self.depth = len(dims)
        self.weights, self.biases = [], []

    def add_noise(self, x):
        if x is None:
            return x
        else:
            if self.noise == 'gaussian':
                n = np.random.normal(0, 0.1, (len(x), len(x[0])))
                return x + n
            if 'mask' in self.noise:
                frac = float(self.noise.split('-')[1])
                temp = np.copy(x)
                for i in temp:
                    n = np.random.choice(len(i), round(
                        frac * len(i)), replace=False)
                    i[n] = 0
                return temp
            if self.noise == 'sp':
                pass


    def fit(self, train_x, eval_x=None):
        for i in range(self.depth):
            print('Layer {0}'.format(i + 1))
            if self.noise is None:
                train_x, eval_x = self.run(data_x=train_x, eval_x=eval_x, activation=self.activations[i],
                                           data_x_=train_x, eval_x_=eval_x, hidden_dim=self.dims[i],
                                           epoch=self.epoch[i], loss=self.loss, batch_size=self.batch_size,
                                           lr=self.lr, print_step=self.print_step)
            else:
                temp_train = np.copy(train_x)
                temp_eval = None if eval_x is None else np.copy(eval_x)
                train_x, eval_x = self.run(data_x=self.add_noise(temp_train), eval_x=self.add_noise(temp_eval),
                                           activation=self.activations[i], data_x_=train_x, eval_x_=eval_x,
                                           hidden_dim=self.dims[i], epoch=self.epoch[i], loss=self.loss,
                                           batch_size=self.batch_size, lr=self.lr, print_step=self.print_step)

    def transform(self, data):
        tf.reset_default_graph()
        sess = tf.Session()
        x = tf.constant(data, dtype=tf.float32)
        for w, b, a in zip(self.weights, self.biases, self.activations):
            weight = tf.constant(w, dtype=tf.float32)
            bias = tf.constant(b, dtype=tf.float32)
            layer = tf.matmul(x, weight) + bias
            x = self.activate(layer, a)
        return x.eval(session=sess)

    def fit_transform(self, train_x, eval_x=None):
        self.fit(train_x=train_x, eval_x=eval_x)
        return self.transform(train_x)

    def run(self, data_x, data_x_, eval_x, eval_x_, hidden_dim, activation, loss, lr,
            print_step, epoch, batch_size=100):
        tf.reset_default_graph()
        input_dim = len(data_x[0])
        sess = tf.Session()
        x = tf.placeholder(dtype=tf.float32, shape=[None, input_dim], name='x')
        x_ = tf.placeholder(dtype=tf.float32, shape=[None, input_dim], name='x_')
        encode = {'weights': tf.Variable(tf.truncated_normal([input_dim, hidden_dim], dtype=tf.float32)),
                  'biases': tf.Variable(tf.truncated_normal([hidden_dim], dtype=tf.float32))}
        decode = {'biases': tf.Variable(tf.truncated_normal([input_dim], dtype=tf.float32)),
                  'weights': tf.transpose(encode['weights'])}
        encoded = self.activate(tf.matmul(x, encode['weights']) + encode['biases'], activation)
        decoded = tf.matmul(encoded, decode['weights']) + decode['biases']

        # reconstruction loss
        if loss == 'rmse':
            loss = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(x_, decoded))))
        elif loss == 'cross-entropy':
            loss = -tf.reduce_mean(x_ * tf.log(decoded))
        train_op = tf.train.AdamOptimizer(lr).minimize(loss)

        sess.run(tf.global_variables_initializer())
        for i in range(epoch):
            b_x, b_x_ = utils.get_batch(
                data_x, data_x_, batch_size)
            sess.run(train_op, feed_dict={x: b_x, x_: b_x_})
            if (i + 1) % print_step == 0:
                loss_train = sess.run(loss, feed_dict={x: data_x, x_: data_x_})
                if eval_x is None:
                    print('epoch {0}: global train loss = {1}'.format(i, loss_train))
                else:
                    loss_eval = sess.run(loss, feed_dict={x: eval_x, x_: eval_x_})
                    print('epoch {0}: global train loss = {1}, global evaluation loss = {2}'
                          .format(i, loss_train, loss_eval))


        self.loss_val = loss_train
        # debug
        # print('Decoded', sess.run(decoded, feed_dict={x: self.data_x_})[0])
        self.weights.append(sess.run(encode['weights']))
        self.biases.append(sess.run(encode['biases']))
        if eval_x is not None:
            eval_x = sess.run(encoded, feed_dict={x: eval_x})
        return sess.run(encoded, feed_dict={x: data_x_}), eval_x

    def activate(self, linear, name):
        if name == 'sigmoid':
            return tf.nn.sigmoid(linear, name='encoded')
        elif name == 'softmax':
            return tf.nn.softmax(linear, name='encoded')
        elif name == 'linear':
            return linear
        elif name == 'tanh':
            return tf.nn.tanh(linear, name='encoded')
        elif name == 'relu':
            return tf.nn.relu(linear, name='encoded')
