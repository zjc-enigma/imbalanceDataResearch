from sklearn.datasets import make_moons
from sklearn.utils import check_random_state
from sklearn.model_selection import train_test_split
import numpy as np





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
