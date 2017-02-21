import numpy as np
import pandas as pd
from sklearn.svm import OneClassSVM
#from mkdata import X_train, X_test, y_train, y_test
from mkdata import tx_train, tx_label_train, tx_test, tx_label_test
import pickle


X_train = tx_train[np.where(tx_label_train==1)]
X_test = tx_test
labels = tx_label_test.values

svm = OneClassSVM(kernel='rbf',
                  gamma=1.0/X_train.shape[0],
                  tol=0.001,
                  nu=0.5,
                  shrinking=True,
                  cache_size=80)

svm = svm.fit(X_train)

#svm = OneClassSVM(kernel='rbf', gamma=1.0/df.shape[0], tol=0.001, nu=0.5, shrinking=True, cache_size=80)
#svm = svm.fit(df.values)
#pickle.dump(svm, open('ocsvm_on_fake_data.model', 'wb'))

scores = svm.decision_function(X_test).flatten()
maxvalue = np.max(scores)
scores = maxvalue - scores

output = pd.DataFrame()

# perform reverse sort
sort_ix = np.argsort(scores)[::-1]

output['labels'] =  labels[sort_ix]
output['scores'] =  scores[sort_ix]

output.to_csv('labels_vs_scores.csv', header=None, index=None, sep="\t")



# plot bound

import matplotlib.pyplot as plt

#plt.figure(1)

xx, yy = np.meshgrid(np.linspace(-3, 3, 300), np.linspace(-3, 3, 300))
Z = svm.decision_function(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
plt.contourf(xx, yy, Z, levels=np.linspace(Z.min(), 0, 7), cmap=plt.cm.PuBu)
a = plt.contour(xx, yy, Z, levels=[0], linewidths=2, colors='darkred')
plt.contourf(xx, yy, Z, levels=[0, Z.max()], colors='palevioletred')


#Z = svm.decision_function(np.c_[X_train.ravel(), y_train.ravel()])
#Z = Z.reshape(X_train.shape)
#legend1["ocsvm"] = plt.contour(X_train, y_train, levels=[0], linewidths=2, colors='b')
# positive_idx = np.where(y_train == 1)
# positive_points = X_train[positive_idx]
# negative_points = X_train[!positive_idx]

s = 50
b1 = plt.scatter(X_imba[y_imba == 0, 0], X_imba[y_imba == 0, 1], color='blue', s=s)
b2 = plt.scatter(X_imba[y_imba == 1, 0], X_imba[y_imba == 1, 1], color='red', s=s)
plt.axis('tight')
plt.xlim((-3, 3))
plt.ylim((-3, 3))

plt.show()
