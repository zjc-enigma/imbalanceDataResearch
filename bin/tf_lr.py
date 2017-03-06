'''
A logistic regression learning algorithm example using TensorFlow library.
This example is using the MNIST database of handwritten digits
(http://yann.lecun.com/exdb/mnist/)

Author: Aymeric Damien
Project: https://github.com/aymericdamien/TensorFlow-Examples/
'''

from __future__ import print_function
#from mkdata import tx_onehot_encoded_df, tx_onehot_encoded_labels
from mkdata import tx_train, tx_test, tx_label_train, tx_label_test
import tensorflow as tf
from sklearn import preprocessing
import pandas as pd
import numpy as np

enc = preprocessing.OneHotEncoder()
tx_label_train = enc.fit_transform(pd.DataFrame(tx_label_train))
tx_label_test = enc.fit_transform(pd.DataFrame(tx_label_test))


# Import MNIST data
#from tensorflow.examples.tutorials.mnist import input_data
#mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

# Parameters
learning_rate = 0.01
training_epochs = 30
batch_size = 1000
display_step = 1


fea_vec_num = tx_train.shape[1]
label_vec_num = 2
train_num = tx_train.shape[0]
test_num = tx_test.shape[0]

# tf Graph Input
x = tf.placeholder(tf.float32, [None, fea_vec_num]) # mnist data image of shape 28*28=784
y = tf.placeholder(tf.float32, [None, label_vec_num]) # 0-9 digits recognition => 10 classes

# Set model weights
W = tf.Variable(tf.zeros([fea_vec_num, label_vec_num]))
b = tf.Variable(tf.zeros([label_vec_num]))

#pred = tf.nn.softmax(tf.matmul(x, W) + b) # Softmax
pred = tf.sigmoid(tf.matmul(x, W) + b) # Sigmoid

# Minimize error using cross entropy
cost = tf.reduce_mean(-tf.reduce_sum(y*tf.log(pred), reduction_indices=1))
# Gradient Descent
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

# Initializing the variables
init = tf.global_variables_initializer()

# Launch the graph
#with tf.Session() as sess:
sess = tf.Session()
sess.run(init)

# Training cycle
for epoch in range(training_epochs):
    avg_cost = 0.
    total_batch = int(train_num/batch_size)
    # Loop over all batches
    for i in range(total_batch):
        begin_idx = batch_size*i
        batch_xs = tx_train[begin_idx:begin_idx+batch_size,  ]
        batch_ys = tx_label_train[begin_idx:begin_idx+batch_size, ]
        # Run optimization op (backprop) and cost op (to get loss value)
        _, c = sess.run([optimizer, cost], feed_dict={x: batch_xs.toarray(), y: batch_ys.toarray()})
        # Compute average loss
        avg_cost += c / total_batch

    #break
    # Display logs per epoch step
    if (epoch+1) % display_step == 0:
        print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(avg_cost))

print("Optimization Finished!")

# with tf.device('/cpu:0'):
# Test model
total_batch = int(test_num/batch_size)

#correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
# Calculate accuracy
#accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
#avg_accuracy = 0.
total_sum = 0.
total_count = 0
pred_res = np.array([])

for i in range(total_batch):
    begin_idx = batch_size*i
    batch_xs = tx_test[begin_idx:begin_idx+batch_size,  ]
    batch_ys = tx_label_test[begin_idx:begin_idx+batch_size, ]
    #a = sess.run(accuracy.eval, feed_dict={x: batch_xs.toarray(), y: batch_ys.toarray()})
    pred_array = sess.run(pred, feed_dict={x: batch_xs.toarray(), y:batch_ys.toarray()})
    pred_cast_array = np.argmax(pred_array)
    correct_prediction = tf.equal(pred_array, tf.argmax(y, 1))
    batch_sum = sess.run(tf.reduce_sum(tf.cast(correct_prediction, tf.float32)),
                         feed_dict={x: batch_xs.toarray(), y:batch_ys.toarray()})
    # batch_count = sess.run(tf.shape(correct_prediction),
    #                        feed_dict={x: batch_xs.toarray(), y:batch_ys.toarray()})[0]
    total_sum += batch_sum
    total_count += batch_size
    pred_res = np.append(pred_res, pred_cast_array)

print("Accuracy:", total_sum/total_count)
#print("Accuracy:", accuracy.eval({x: tx_test.toarray(), y: tx_label_test.toarray()}))
#sess.run([pred, tf] )
