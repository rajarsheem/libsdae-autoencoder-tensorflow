import tensorflow as tf

from data import get_batch, get_full


class BasicAutoEncoder:
    """A basic autoencoder with a single hidden layer"""

    def __init__(self, input_dim, hidden_dim, epoch=1000, batch_size=50):
        self.batch_size = batch_size
        self.epoch = epoch
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        self.hidden_feature = []

    def forward(self, x):
        with tf.name_scope('encode'):
            weights = tf.Variable(tf.random_normal([self.input_dim, self.hidden_dim], dtype=tf.float32),
                                  name='weights')
            biases = tf.Variable(tf.zeros([self.hidden_dim]), name='biases')
            encoded = tf.nn.sigmoid(tf.matmul(x, weights) + biases,name='encoded')

        with tf.name_scope('decode'):
            weights = tf.Variable(tf.random_normal([self.hidden_dim, self.input_dim], dtype=tf.float32),
                                  name='weights')
            biases = tf.Variable(tf.zeros([self.input_dim]), name='biases')
            decoded = tf.matmul(encoded, weights) + biases
        # loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(decoded, x, name='cross_entropy'))
        return encoded, decoded

    def train(self, x, decoded):
        loss = tf.sqrt(tf.reduce_mean(tf.square(tf.sub(x, decoded))))
        train_op = tf.train.AdamOptimizer(0.001).minimize(loss)
        return loss, train_op

    def run(self):
        with tf.Graph().as_default():
            x = tf.placeholder(dtype=tf.float32, shape=[None, self.input_dim], name='x')
            encoded, decoded = self.forward(x)
            loss, train_op = self.train(x, decoded)
            with tf.Session() as sess:
                sess.run(tf.initialize_all_variables())
                for i in range(self.epoch):
                    for j in range(50):
                        b_x = get_batch(self.batch_size)
                        l, _ = sess.run([loss, train_op], feed_dict={x: b_x})
                    if i % 100 == 0:
                        print(l)
                self.hidden_feature = sess.run(encoded, feed_dict={x: get_full()})

    def get_hidden_feature(self):
        return self.hidden_feature

ae = BasicAutoEncoder(4, 6)
ae.run()
h = ae.get_hidden_feature()
print(h.shape)