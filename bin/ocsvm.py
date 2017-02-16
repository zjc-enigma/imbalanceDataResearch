import numpy as np
import pandas as pd
from sklearn.svm import OneClassSVM
from mkdata import X_train, X_test, y_train, y_test

df = pd.DataFrame(X_train)
labels = pd.Series(y_train)

svm = OneClassSVM(kernel='rbf', gamma=1.0/df.shape[0], tol=0.001, nu=0.5, shrinking=True, cache_size=80)
svm = svm.fit(df.values)

scores = svm.decision_function(df.values).flatten()
maxvalue = np.max(scores)
scores = maxvalue - scores

output = pd.DataFrame()

# perform reverse sort
sort_ix = np.argsort(scores)[::-1]

output['labels'] =  labels[sort_ix]
output['scores'] =  scores[sort_ix]

output.to_csv('labels_vs_scores.csv', header=None, index=None, sep="\t")
