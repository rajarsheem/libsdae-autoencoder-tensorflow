import numpy as np
import deepautoencoder.utils as utils
import tensorflow as tf
from sklearn.metrics import r2_score
import os

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
        self.weights, self.biases, self.dec_biases = [], [], []
        self.trained_model = False



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


    def fit(self, train_x, val_x=None):

        self.weights.clear()
        self.biases.clear()
        self.dec_biases.clear()

        for i in range(self.depth):
            print('Layer {0}'.format(i + 1))
            if self.noise is None:
                train_x, val_x = self.run(data_x=train_x, val_x=val_x, activation=self.activations[i],
                                           data_x_=train_x, val_x_=val_x, hidden_dim=self.dims[i],
                                           epoch=self.epoch[i], loss=self.loss, batch_size=self.batch_size,
                                           lr=self.lr, print_step=self.print_step, depth=i+1)
            else:
                temp_train = np.copy(train_x)
                temp_val = None if val_x is None else np.copy(val_x)
                train_x, val_x = self.run(data_x=self.add_noise(temp_train), val_x=self.add_noise(temp_val),
                                           activation=self.activations[i], data_x_=train_x, val_x_=val_x,
                                           hidden_dim=self.dims[i], epoch=self.epoch[i], loss=self.loss,
                                           batch_size=self.batch_size, lr=self.lr, print_step=self.print_step, depth=i+1)
        self.trained_model = True


    def partial_fit(self, train_x, val_x=None):

        self.weights.clear()
        self.biases.clear()
        self.dec_biases.clear()

        if self.trained_model:
            for i in range(self.depth):
                print('Layer {0}'.format(i + 1))
                if self.noise is None:
                    train_x, val_x = self.partial_run(data_x=train_x, val_x=val_x, activation=self.activations[i],
                                                      data_x_=train_x, val_x_=val_x, hidden_dim=self.dims[i],
                                                      epoch=self.epoch[i], loss=self.loss, batch_size=self.batch_size,
                                                      lr=self.lr, print_step=self.print_step, depth=i + 1)
                else:
                    temp_train = np.copy(train_x)
                    temp_val = None if val_x is None else np.copy(val_x)
                    train_x, val_x = self.partial_run(data_x=self.add_noise(temp_train), val_x=self.add_noise(temp_val),
                                                      activation=self.activations[i], data_x_=train_x, val_x_=val_x,
                                                      hidden_dim=self.dims[i], epoch=self.epoch[i], loss=self.loss,
                                                      batch_size=self.batch_size, lr=self.lr,
                                                      print_step=self.print_step, depth=i + 1)
        else:
            # model must be trained once, before partial_fit
            self.fit(train_x, val_x)


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

    def getReconsturction(self, data):
        tf.reset_default_graph()
        sess = tf.Session()
        x = tf.constant(data, dtype=tf.float32)
        for w, b, a in zip(self.weights, self.biases, self.activations):
            weight = tf.constant(w, dtype=tf.float32)
            bias = tf.constant(b, dtype=tf.float32)
            layer = tf.matmul(x, weight) + bias
            x = self.activate(layer, a)

        for w, b, a in zip(reversed(self.weights), reversed(self.dec_biases), reversed(self.activations)):
            weight = tf.constant(w, dtype=tf.float32)
            weight = tf.transpose(weight)
            bias = tf.constant(b, dtype=tf.float32)
            layer = tf.matmul(x, weight) + bias
            x = self.activate(layer, a)


        return x.eval(session=sess)


    def fit_transform(self, train_x, val_x=None):
        self.fit(train_x=train_x, val_x=val_x)
        return self.transform(train_x)

    def run(self, data_x, data_x_, val_x, val_x_, hidden_dim, activation, loss, lr,
            print_step, epoch, batch_size=100, depth=0):
        tf.reset_default_graph()
        input_dim = len(data_x[0])
        sess = tf.Session()
        x = tf.placeholder(dtype=tf.float32, shape=[None, input_dim], name='x')
        x_ = tf.placeholder(dtype=tf.float32, shape=[None, input_dim], name='x_')
        encode = {'weights': tf.Variable(tf.truncated_normal([input_dim, hidden_dim], dtype=tf.float32), name='enc_weights'),
                  'biases': tf.Variable(tf.truncated_normal([hidden_dim], dtype=tf.float32), name='enc_biases')}
        decode = {'biases': tf.Variable(tf.truncated_normal([input_dim], dtype=tf.float32), name='dec_biases'),
                  'weights': tf.transpose(encode['weights'], name='dec_weights')}
        encoded = self.activate(tf.matmul(x, encode['weights']) + encode['biases'], activation)
        decoded = tf.add(tf.matmul(encoded, decode['weights']), decode['biases'], name='decoded')

        # reconstruction loss
        if loss == 'rmse':
            loss = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(x_, decoded))), name='loss')
        elif loss == 'cross-entropy':
            loss = tf.negative(tf.reduce_mean(x_ * tf.log(decoded), name='loss'), name='loss')

        train_op = tf.train.AdamOptimizer(lr).minimize(loss)
        saver = tf.train.Saver()


        sess.run(tf.global_variables_initializer())
        for i in range(epoch):
            b_x, b_x_ = utils.get_batch(
                data_x, data_x_, batch_size)
            sess.run(train_op, feed_dict={x: b_x, x_: b_x_})
            if (i + 1) % print_step == 0:
                loss_train = sess.run(loss, feed_dict={x: data_x, x_: data_x_})
                dec_train = sess.run(decoded, feed_dict={x: data_x})
                if val_x is None:
                    print('epoch {0}: train loss = {1:.5f}, R²-Score Train = {2:.3f}'
                          .format(i, loss_train, r2_score(data_x_, dec_train)))
                else:
                    loss_val = sess.run(loss, feed_dict={x: val_x, x_: val_x_})
                    dec_val = sess.run(decoded, feed_dict={x: val_x})
                    print('epoch {0}: train loss = {1:.5f}, validation loss = {2:.5f},'
                          ' R²-Score Train = {3:.3f}, R²-Score Val = {4:.3f}'
                          .format(i, loss_train, loss_val, r2_score(data_x_, dec_train,
                                                                    multioutput='variance_weighted'),
                                  r2_score(val_x_, dec_val, multioutput='variance_weighted')))




        self.loss_val = loss_train
        # debug
        # print('Decoded', sess.run(decoded, feed_dict={x: self.data_x_})[0])
        self.weights.append(sess.run(encode['weights']))
        self.biases.append(sess.run(encode['biases']))
        self.dec_biases.append(sess.run(decode['biases']))
        if val_x is not None:
            val_x = sess.run(encoded, feed_dict={x: val_x_})
        train_x = sess.run(encoded, feed_dict={x: data_x_})
        tf.add_to_collection("train_op", train_op)
        layerPath = './SDA_model/Layer{}/SDA_model'.format(depth)
        if not os.path.exists(layerPath):
            os.makedirs(layerPath)
        save_path = saver.save(sess, layerPath)
        print("SDA-Model saved in path: %s" % save_path)
        return train_x, val_x



    def partial_run(self, data_x, data_x_, val_x, val_x_, hidden_dim, activation, loss, lr,
            print_step, epoch, batch_size=100, depth=0):

        # TODO: include this method to run with a additional argument: restore_model = True

        tf.reset_default_graph()
        sess = tf.Session()
        saver = tf.train.import_meta_graph('./SDA_model/Layer{}/SDA_model.meta'.format(depth))
        saver.restore(sess, tf.train.latest_checkpoint('./SDA_model/Layer{}/'.format(depth)))

        graph = tf.get_default_graph()
        x = graph.get_tensor_by_name("x:0")
        x_ = graph.get_tensor_by_name("x_:0")
        decoded = graph.get_tensor_by_name("decoded:0")
        encoded = graph.get_tensor_by_name("encoded:0")
        enc_weights = graph.get_tensor_by_name("enc_weights:0")
        enc_biases = graph.get_tensor_by_name("enc_biases:0")
        dec_biases = graph.get_tensor_by_name("dec_biases:0")
        train_op = tf.get_collection("train_op")[0]
        loss = graph.get_tensor_by_name("loss:0")


        sess.run(tf.global_variables_initializer())
        for i in range(epoch):
            b_x, b_x_ = utils.get_batch(
                data_x, data_x_, batch_size)
            sess.run(train_op, feed_dict={x: b_x, x_: b_x_})
            if (i + 1) % print_step == 0:
                loss_train = sess.run(loss, feed_dict={x: data_x, x_: data_x_})
                dec_train = sess.run(decoded, feed_dict={x: data_x})
                if val_x is None:
                    print('epoch {0}: train loss = {1:.5f}, R²-Score Train = {2:.3f}'
                          .format(i, loss_train, r2_score(data_x_, dec_train)))
                else:
                    loss_val = sess.run(loss, feed_dict={x: val_x, x_: val_x_})
                    dec_val = sess.run(decoded, feed_dict={x: val_x})
                    print('epoch {0}: train loss = {1:.5f}, validation loss = {2:.5f},'
                          ' R²-Score Train = {3:.3f}, R²-Score Val = {4:.3f}'
                          .format(i, loss_train, loss_val, r2_score(data_x_, dec_train,
                                                                    multioutput='variance_weighted'),
                                  r2_score(val_x_, dec_val, multioutput='variance_weighted')))




        self.loss_val = loss_train
        # debug
        # print('Decoded', sess.run(decoded, feed_dict={x: self.data_x_})[0])
        self.weights.append(sess.run(enc_weights))
        self.biases.append(sess.run(enc_biases))
        self.dec_biases.append(sess.run(dec_biases))
        if val_x is not None:
            val_x = sess.run(encoded, feed_dict={x: val_x_})
        train_x = sess.run(encoded, feed_dict={x: data_x_})
        tf.add_to_collection("train_op", train_op)
        save_path = saver.save(sess, './SDA_model/Layer{}/SDA_model'.format(depth))
        print("SDA-Model saved in path: %s" % save_path)
        return train_x, val_x

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
