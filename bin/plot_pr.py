import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import auc


ocsvm_pr = pd.read_csv('pr.txt', header=None, index_col=False, skiprows=1)
ocsvm_auc = auc(ocsvm_pr[0], ocsvm_pr[1])

fig = plt.figure(figsize=(12,5))
ax = fig.add_axes([0.045, 0.1, 0.6, 0.8])
ax.plot(ocsvm_pr[0].values, ocsvm_pr[1].values, label='one_class        AUC=%f' % ocsvm_auc, lw=2)
plt.show()
