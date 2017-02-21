from sklearn.datasets import make_moons
from sklearn.utils import check_random_state
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.feature_extraction import FeatureHasher
import numpy as np
import pandas as pd




X, y = make_moons(n_samples=500000,
                  shuffle=True,
                  noise=0.5,
                  random_state=10)

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



X_train, X_test, y_train, y_test = train_test_split(X[idx], y[idx],
                                                    test_size=0.33,
                                                    random_state=42)

tx_df = pd.read_csv("../data/tx_sample", sep="\t",
                      header=None,
                      error_bad_lines=False)
tx_headers = pd.read_csv("../data/tx_sample_desc", sep=", ",
                      header=None)


tx_df.columns = tx_headers.values[0]

tx_labels = tx_df.label

# TODO device os optim
# tx_fea_df = tx_df[[
#     'IdAdUnitId',
# ]]

tx_fea_df = tx_df[[
    'ActionPlatform',
    'AgentAppId',
    'DeviceOs',
    'DeviceBrand',
    'DeviceModel',
    'IdAdUnitId',
    'company',
    'GeoId',
    'CreativeId',
    'width',
    'height']]

# convert all int column to string
tx_fea_df = tx_fea_df.applymap(str)

lenc = preprocessing.LabelEncoder()
tx_label_encoded_df = tx_fea_df.apply(lenc.fit_transform, axis=0)

enc = preprocessing.OneHotEncoder()
tx_onehot_encoded_df = enc.fit_transform(tx_label_encoded_df)

tx_train, tx_test, tx_label_train, tx_label_test = train_test_split(
    tx_onehot_encoded_df,
    tx_labels,
    test_size=0.33,
    random_state=42)



tx_onehot_encoded_labels = enc.fit_transform(pd.DataFrame(tx_labels))
