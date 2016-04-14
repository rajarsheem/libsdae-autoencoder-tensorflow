import tensorflow as tf
import deepautoencoder.data


class BasicAutoEncoder:
    """A basic autoencoder with a single hidden layer. This is not to be externally used but internally by
    StackedAutoEncoder"""

    def __init__(self, data_x, data_x_, hidden_dim, activation, loss, lr, print_step, epoch, batch_size=50):
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
        self.encoded = None
        self.decoded = None
        self.x = None
        self.x_ = None

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

    def train(self, x_, decoded):
        if self.loss == 'rmse':
            loss = tf.sqrt(tf.reduce_mean(tf.square(tf.sub(x_, decoded))))
        else:
            loss = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(decoded, x_))
        train_op = tf.train.AdamOptimizer(self.lr).minimize(loss)
        return loss, train_op

    def run(self):
        sess = tf.Session()
        self.x = tf.placeholder(dtype=tf.float32, shape=[None, self.input_dim], name='x')
        self.x_ = tf.placeholder(dtype=tf.float32, shape=[None, self.input_dim], name='x_')
        encode = {'weights': tf.Variable(tf.truncated_normal([self.input_dim, self.hidden_dim], dtype=tf.float32)),
                  'biases': tf.Variable(tf.truncated_normal([self.hidden_dim], dtype=tf.float32))}
        encoded_vals = tf.matmul(self.x, encode['weights']) + encode['biases']
        self.encoded = self.activate(encoded_vals, self.activation)
        decode = {'biases': tf.Variable(tf.truncated_normal([self.input_dim], dtype=tf.float32))}
        self.decoded = tf.matmul(self.encoded, tf.transpose(encode['weights'])) + decode['biases']
        loss, train_op = self.train(self.x_, self.decoded)
        sess.run(tf.initialize_all_variables())
        for i in range(self.epoch):
            b_x, b_x_ = deepautoencoder.data.get_batch(self.data_x, self.data_x_, self.batch_size)
            sess.run(train_op, feed_dict={self.x: b_x, self.x_: b_x_})
            if (i + 1) % self.print_step == 0:
                l = sess.run(loss, feed_dict={self.x: self.data_x, self.x_: self.data_x_})
                print('epoch {0}: global loss = {1}'.format(i, l))
                # print(sess.run(encode['weights'])[0])
        # debug
        # print('Decoded', sess.run(self.decoded, feed_dict={self.x: self.data_x_})[0])
        return sess.run(self.encoded, feed_dict={self.x: self.data_x_}), sess.run(
            encode['weights']), sess.run(encode['biases'])
