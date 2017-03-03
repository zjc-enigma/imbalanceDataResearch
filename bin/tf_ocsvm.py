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



fea_vec_num = tx_train.shape[1]
label_vec_num = 2
train_num = tx_train.shape[0]
test_num = tx_test.shape[0]


x = tf.placeholder(tf.float32, [None, fea_vec_num])
y = tf.placeholder(tf.float32, [None, label_vec_num])

W = tf.Variable(tf.zeros([fea_vec_num, label_vec_num]))
b = tf.Variable(tf.zeros([label_vec_num]))
