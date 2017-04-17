import sys
sys.path.insert(0, '../source')
import tensorflow as tf
from tensorflow.contrib import rnn
from pymongo import MongoClient
import get_data
import numpy as np
from tensorflow.python import debug as tf_debug


# Parameters
learning_rate = 0.001
training_iters = 5000
display_step = 1000
input_dimension = 53
output_dimension = 2
n_hidden = 256
n_input = 5
batch_size = 32

x = tf.placeholder(tf.float32, [None, n_input, input_dimension])
y = tf.placeholder(tf.float32, [None, output_dimension])


# RNN output node weights and biases
weights = {
    'out': tf.Variable(tf.random_normal([n_hidden, output_dimension]))
}
biases = {
    'out': tf.Variable(tf.random_normal([output_dimension]))
}


def RNN(x, weights, biases):
    rnn_cell = rnn.BasicLSTMCell(n_hidden)
    rnn_cell = rnn.MultiRNNCell([rnn_cell] * 2)
    rnn_cell = rnn.DropoutWrapper(rnn_cell, 0.8)
    x = tf.unstack(x, n_input, 1)
    outputs, states = rnn.static_rnn(rnn_cell, x, dtype=tf.float32)
    # there are n_input outputs but
    # we only want the last output
    return tf.matmul(outputs[-1], weights['out']) + biases['out']

pred = RNN(x, weights, biases)

# Loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate).minimize(cost)
# Model evaluation
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(pred, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# Initializing the variables
init = tf.global_variables_initializer()


period_data = get_data.get_period_data(n_input, 'tweets', 'tweets_10_periods')
input_data, output_data = get_data.input_output_generation(period_data, n_input, 'users', 'user-profiling', 0)
train_input, train_output, test_input, test_output = get_data.split_data_to_train_test(input_data, output_data, n_input)
with tf.Session() as session:
    # session = tf_debug.LocalCLIDebugWrapperSession(session)
    # session.add_tensor_filter("has_inf_or_nan", tf_debug.has_inf_or_nan)
    session.run(init)
    step = 0
    loss_total = 0
    acc_total = 0
    offset = 0
    while step < training_iters:
        input_batch, output_batch = get_data.get_batch(train_input, train_output, batch_size)
        _, acc, loss, prediction = session.run([optimizer, accuracy, cost, pred], \
                                                feed_dict={x: input_batch, y: output_batch})

        print("Iter= " + str(step + 1) + ", Average Loss= " + \
              "{:.6f}".format(loss) + ", Average Accuracy= " + \
              "{:.2f}%".format(100 * acc))
        step += 1

    incorrect = session.run(accuracy,{x: test_input, y: test_output})
    print('Epoch {:2d} error {:3.1f}%'.format(training_iters + 1, 100 * incorrect))
    session.close()