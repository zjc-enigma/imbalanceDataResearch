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


def cost():
    pass



def error():
    pass

