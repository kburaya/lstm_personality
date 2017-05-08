import sys

sys.path.insert(0, '../source')
import tensorflow as tf
from tensorflow.contrib import rnn
import get_data
import logging

class LSTM_model:
    def __init__(self, learning_rate, training_iters, display_step, input_dimension, output_dimension,
                 n_hidden, n_input, batch_size, label, multi_layer=False):
        # technical detales
        self.logger = logging.getLogger('LSTM_logger')
        self.MBTI_labels = ['E_I', 'S_N', 'T_F', 'J_P']

        self.learning_rate = learning_rate
        self.training_iters = training_iters
        self.display_step = display_step
        self.input_dimension = input_dimension  # number of features
        self.output_dimension = output_dimension  # number of labels
        self.n_hidden = n_hidden
        self.n_input = n_input  # number of time periods
        self.batch_size = batch_size
        self.multi_layer = multi_layer
        self.label = label

        self.logger.info('Model for %s label' % self.MBTI_labels[label])

        self.x = tf.placeholder(tf.float32, [None, self.n_input, self.input_dimension])
        self.y = tf.placeholder(tf.float32, [None, self.output_dimension])

        # RNN output node weights and biases
        self.weights = {
            'out': tf.Variable(tf.random_normal([self.n_hidden, self.output_dimension]))
        }
        self.biases = {
            'out': tf.Variable(tf.random_normal([self.output_dimension]))
        }
        self.pred, self.cost, self.optimizer, self.accuracy = self.set_optimizers()

    def RNN(self):
        rnn_cell = rnn.BasicLSTMCell(self.n_hidden)
        if self.multi_layer:
            rnn_cell = rnn.MultiRNNCell([rnn_cell] * 2)
        rnn_cell = rnn.DropoutWrapper(rnn_cell, 0.8)
        x = tf.unstack(self.x, self.n_input, 1)
        outputs, states = rnn.static_rnn(rnn_cell, x, dtype=tf.float32)
        # there are n_input outputs but
        # we only want the last output
        return tf.matmul(outputs[-1], self.weights['out']) + self.biases['out']

    def set_optimizers(self):
        pred = self.RNN()
        # Loss and optimizer
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=self.y))
        optimizer = tf.train.RMSPropOptimizer(learning_rate=self.learning_rate).minimize(cost)
        # Model evaluation
        correct_prediction = tf.equal(tf.argmax(self.y, 1), tf.argmax(pred, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        return pred, cost, optimizer, accuracy

    def train(self, train_input, train_output, batch_size, test_input, test_output):
        self.logger.info('Train model with: learning_rate=%s\n'
                         'training_iters=%d\n'
                         'n_hidden=%d\n'
                         'time_periods=%d\n'
                         'batch_size=%d\n'
                         'is_multi_layer=%s\n'
                         'label=%s' % (str(self.learning_rate), self.training_iters, self.n_hidden, self.n_input,
                                       self.batch_size, str(self.multi_layer), self.MBTI_labels[self.label]))
        init = tf.global_variables_initializer()
        with tf.Session() as session:
            session.run(init)
            step = 0
            while step < self.training_iters:
                input_batch, output_batch = get_data.get_batch(train_input, train_output, batch_size)
                _, acc, loss, prediction = session.run([self.optimizer, self.accuracy, self.cost, self.pred], \
                                                       feed_dict={self.x: input_batch, self.y: output_batch})
                if step % self.display_step == 0:
                    self.logger.info("Iter= " + str(step + 1) + ", Average Loss= " + \
                                     "{:.6f}".format(loss) + ", Average Accuracy= " + \
                                     "{:.2f}%".format(100 * acc))
                step += 1

            _, acc, loss, prediction = session.run([self.optimizer, self.accuracy, self.cost, self.pred],
                                                   {self.x: test_input, self.y: test_output})
            self.logger.info(
                'Epoch {:2d} Average Accuracy on test set {:3.1f}%'.format(self.training_iters + 1, 100 * acc))
            session.close()
