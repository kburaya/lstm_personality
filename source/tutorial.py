import sys
sys.path.insert(0, '../source')
import tensorflow as tf
from tensorflow.contrib import rnn
from pymongo import MongoClient
import get_data
import numpy as np

###
MONGO_PORT = 27017
MONGO_HOST = '172.29.29.24'
MONGO_DB_NAME = 'user-profiling'
###

# Parameters
learning_rate = 0.001
training_iters = 50000
display_step = 1000
batch_size = 2
features_dimensions = 53
# number of units in RNN cell
n_hidden = 512



x = tf.placeholder(tf.float32, [None, batch_size, features_dimensions])
y = tf.placeholder(tf.float32, [None, 2])


# RNN output node weights and biases
weights = {
    'out': tf.Variable(tf.random_normal([n_hidden, 2]))
}
biases = {
    'out': tf.Variable(tf.random_normal([2]))
}


def RNN(x, weights, biases):
    rnn_cell = rnn.BasicLSTMCell(n_hidden)
    # Generate a n_input-element sequence of inputs
    # (eg. [had] [a] [general] -> [20] [6] [33])
    # generate prediction
    outputs, states = tf.nn.dynamic_rnn(rnn_cell, x, dtype=tf.float32)
    # there are n_input outputs but
    # we only want the last output
    return tf.matmul(outputs[-1], weights['out']) + biases['out']

pred = RNN(x, weights, biases)
print (pred)

# Loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate).minimize(cost)

# Model evaluation
correct_pred = tf.equal(pred, y)
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initializing the variables
init = tf.global_variables_initializer()


period_data = get_data.get_period_data(2, 'tweets', 'tweets_10_periods')
input_data, output_data = get_data.input_output_generation(period_data, 'users', 'user-profiling', 1)
train_input, train_output, test_input, test_output = get_data.split_data_to_train_test(input_data, output_data)
with tf.Session() as session:
    session.run(init)
    step = 0
    loss_total = 0
    acc_total = 0
    while step < training_iters:
        _, acc, loss, pred = session.run([optimizer, accuracy, cost, pred], \
                                                feed_dict={x: train_input, y: train_output})

        print("Iter= " + str(step + 1) + ", Average Loss= " + \
              "{:.6f}".format(loss) + ", Average Accuracy= " + \
              "{:.2f}%".format(100 * acc))

    incorrect = session.run(accuracy,{x: test_input, y: test_output})
    print('Epoch {:2d} error {:3.1f}%'.format(training_iters + 1, 100 * incorrect))
    session.close()