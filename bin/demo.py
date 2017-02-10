#!/usr/bin/python

import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import seaborn as sns
from sklearn.pipeline import Pipeline
from sklearn import linear_model
from sklearn.datasets import make_moons
from imblearn.datasets import make_imbalance
from sklearn.utils import check_random_state, check_X_y
import pandas as pd

from sklearn.svm import OneClassSVM
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier

#proba_test = clf.predict_proba(X_test)
from sklearn.metrics import accuracy_score
from sklearn.metrics import auc
from sklearn.metrics import roc_curve

import random
from sklearn.model_selection import train_test_split

matplotlib.use('Agg')
sns.set()
almost_black = '#262626'
palette = sns.color_palette()

X, y = make_moons(n_samples=500000,
                  shuffle=True,
                  noise=0.5,
                  random_state=10)

# make training set
#X_, y_ = make_imbalance(X, y, ratio=ratio, min_c_=1)

df_origin = pd.DataFrame(X)
df_origin['y'] = y


min_c_ = 1
ratio = 0.001
mask = y == min_c_

random_state = None
random_state = check_random_state(random_state)
n_min_samples = int(np.count_nonzero(y != min_c_) * ratio)

idx_maj = np.where(~mask)[0]
idx_min = np.where(mask)[0]
idx_min = random_state.choice(idx_min, size=n_min_samples, replace=False)
idx = np.concatenate((idx_min, idx_maj), axis=0)

#df_imbalanced = df_origin.iloc[idx]
#train_idx = df_origin.index.isin(idx)

X_train, X_test, y_train, y_test = train_test_split(X[idx],
                                                    y[idx],
                                                    test_size=0.33,
                                                    random_state=42)

# df_test = df_origin[~train_idx]
# X_train = df_train.ix[:, :2]
# y_train = df_train['y']
# X_test = df_test.ix[:, :2]
# y_test = df_test['y']

X_ = X_train
y_ = y_train
# plot & show
fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(X_[y_ == 0, 0],
           X_[y_ == 0, 1],
           label="Class #0",
           alpha=0.5,
           edgecolor=almost_black,
           facecolor=palette[0],
           linewidth=0.15)

ax.scatter(X_[y_ == 1, 0],
           X_[y_ == 1, 1],
           label="Class #1",
           alpha=0.5,
           edgecolor=almost_black,
           facecolor=palette[2],
           linewidth=0.15)

plt.tight_layout()
plt.show()

def evaluate_auc(model, test_data, test_real):
    pred = model.predict_proba(test_data)[:, 1]
    fpr, tpr, thresh = roc_curve(test_real, pred, pos_label=1)
    print("auc :", auc(fpr, tpr))



# using lr as baseline
# using X_ y_ as training data, the X, y exclude X_, y_ as testing data
lr = linear_model.LogisticRegression(solver='lbfgs')
lr_model = lr.fit(X_train, y_train)
evaluate_auc(lr_model, X_test, y_test)



#lr_model.score(X_test, y_test)

# using ocsvm model
ocsvm_clf = OneClassSVM(kernel='linear')

#for i, (clf_name, clf) in enumerate(classifiers.items()):
ocsvm_clf.fit(X_train, y_train)
pred_test = ocsvm_clf.predict(X_test)

fpr, tpr, thresh = roc_curve(y_test, pred_test)

#y_pred_train = clf.predict(X_train)


# clf = OneVsRestClassifier(BaggingClassifier(
#     SVC(kernel='linear', probability=True, class_weight='auto'),
#     max_samples=1.0 / n_estimators, n_estimators = n_estimators), n_jobs=88)

# using oneVSRest model
n_estimators = 10

clf = OneVsRestClassifier(SVC(kernel='linear', probability=True), n_jobs=88)
clf.fit(X_train, y_train)
evaluate_auc(clf, X_test, y_test)


#pred_test = clf.predict_proba(X_test)[:, 1]
#fpr, tpr, thresh = roc_curve(y_test, pred_test, pos_label=1)

