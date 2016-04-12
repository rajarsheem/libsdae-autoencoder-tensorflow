import tensorflow as tf

import deepautoencoder.data


class BasicAutoEncoder:
    """A basic autoencoder with a single hidden layer"""

    def __init__(self, data_x, data_x_, hidden_dim, activation, loss, lr, print_step, epoch=1000, batch_size=50):
        self.print_step = print_step
        self.lr = lr
        self.loss = loss
        self.activation = activation
        self.data_x_ = data_x_
        self.data_x = data_x
        self.batch_size = batch_size
        self.epoch = epoch
        self.hidden_dim = hidden_dim
        self.input_dim = len(data_x[0])
        self.hidden_feature = []
        self.sess = tf.Session()
        self.encoded = None
        self.decoded = None
        self.x = None
        self.x_ = None

    def train(self, x_, decoded):
        if self.loss == 'rmse':
            loss = tf.sqrt(tf.reduce_mean(tf.square(tf.sub(x_, decoded))))
        else:
            loss = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(decoded, x_))
        train_op = tf.train.AdamOptimizer(self.lr).minimize(loss)
        return loss, train_op

    def run(self):
        self.x = tf.placeholder(dtype=tf.float32, shape=[None, self.input_dim], name='x')
        self.x_ = tf.placeholder(dtype=tf.float32, shape=[None, self.input_dim], name='x_')

        encode = {'weights': tf.Variable(tf.random_normal([self.input_dim, self.hidden_dim], dtype=tf.float32)),
                  'biases': tf.Variable(tf.zeros([self.hidden_dim]))}
        encoded_vals = tf.matmul(self.x, encode['weights']) + encode['biases']

        if self.activation == 'sigmoid':
            self.encoded = tf.nn.sigmoid(encoded_vals, name='encoded')
        elif self.activation == 'softmax':
            self.encoded = tf.nn.softmax(encoded_vals, name='encoded')
        elif self.activation == 'linear':
            self.encoded = encoded_vals
        elif self.activation == 'tanh':
            self.encoded = tf.nn.tanh(encoded_vals, name='encoded')

        decode = {'biases': tf.Variable(tf.zeros([self.input_dim]))}
        self.decoded = tf.matmul(self.encoded, tf.transpose(encode['weights'])) + decode['biases']

        loss, train_op = self.train(self.x_, self.decoded)
        self.sess = tf.Session()
        self.sess.run(tf.initialize_all_variables())
        for i in range(self.epoch):
            b_x, b_x_ = deepautoencoder.data.get_batch(self.data_x, self.data_x_, self.batch_size)
            self.sess.run(train_op, feed_dict={self.x: b_x, self.x_: b_x_})
            if (i+1) % self.print_step == 0:
                l = self.sess.run(loss, feed_dict={self.x: self.data_x, self.x_: self.data_x_})
                print('epoch {0}: global loss = {1}'.format(i, l))
        # debug
        # print('Decoded', self.sess.run(self.decoded, feed_dict={self.x: self.data_x_})[0])

        return self.sess.run(self.encoded, feed_dict={self.x: self.data_x_}), self.sess.run(
            encode['weights']), self.sess.run(encode['biases'])

    def transform(self, data):
        temp = self.sess.run(self.encoded, feed_dict={self.x: data})
        return temp
