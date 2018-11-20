import pandas as pd
import numpy as np
from numpy.random import shuffle
from sklearn.preprocessing import minmax_scale

def get_data():
    tes = pd.read_csv("./pendigits.tes").as_matrix()
    tra = pd.read_csv("./pendigits.tra").as_matrix()

    all_data = np.append(tes, tra, axis=0);
    shuffle(all_data)

    X = all_data[:, :16]
    y = all_data[:, 16:]
    X = minmax_scale(X.astype(float))


    X_train = X[:5000]
    y_train = y[:5000].reshape(-1)
    X_validation = X[5000:]
    y_validation = y[5000:].reshape(-1)


    return X_train, X_validation, y_train, y_validation