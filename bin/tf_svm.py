from __future__ import print_function
#from mkdata import tx_onehot_encoded_df, tx_onehot_encoded_labels
from mkdata import tx_train, tx_test, tx_label_train, tx_label_test
import tensorflow as tf
from sklearn import preprocessing
import pandas as pd
import numpy as np
from sklearn import datasets

enc = preprocessing.OneHotEncoder()
tx_label_train = enc.fit_transform(pd.DataFrame(tx_label_train))
tx_label_test = enc.fit_transform(pd.DataFrame(tx_label_test))
iris = datasets.load_iris()
train_X = iris.data[:, :2]
train_y = iris.target
# important! makes data shape correct
train_y = train_y.reshape(train_y.size,1)

BATCH_SIZE = 50
EPOCH_NUM = 5000
LEARNING_RATE = 0.01
# C parameter
SVMC = 1

train_size, feature_num = train_X.shape

X = tf.placeholder("float", shape=[None, feature_num])
y = tf.placeholder("float", shape=[None, 1])


W = tf.Variable(tf.zeros([feature_num, 1]))
b = tf.Variable(tf.zeros([1]))

y_raw = tf.matmul(X, W) + b

# optim
regularization_loss = 0.5*tf.reduce_sum(tf.square(W))
hinge_loss = tf.reduce_sum(tf.maximum(tf.zeros([BATCH_SIZE,1]), 
                                      1 - y*y_raw))


loss_func = regularization_loss + SVMC*hinge_loss
optimizer = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(loss_func)


pred = tf.sign(y_raw)
correct_pred = tf.equal(y, pred)
accuracy = tf.reduce_mean(tf.cast(correct_pred, "float"))
init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init)

with tf.device('/cpu:0'):
    for epoch in range(EPOCH_NUM):
        total_batch = int(train_size / BATCH_SIZE)
        for step in range(total_batch):

            begin_idx = BATCH_SIZE * step
            batch_feas = train_X[begin_idx:(begin_idx + BATCH_SIZE), :]
            batch_labels = train_y[begin_idx:(begin_idx + BATCH_SIZE)]
            sess.run(optimizer, feed_dict={X: batch_feas, y: batch_labels})
            print('loss: ',
                  sess.run(loss_func,
                           feed_dict={
                               X: batch_feas,
                               y: batch_labels}))



sess.close()




