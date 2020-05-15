
# created at 2020.05.15 by Gaofeng

# 在Michael的代码的基础上改了一点,就是下载&读取mnist数据集

# %%
from pathlib import Path
import requests
import pickle
import gzip
import numpy as np


def read_mnist():

    def oneHot(x):
        x_onehot = np.zeros((len(x), 10, 1))
        for index in range(len(x)):
            x_onehot[index, x[index], 0] = 1
        return x_onehot
    DATA_PATH = Path("data")
    PATH = DATA_PATH / "mnist"
    PATH.mkdir(parents=True, exist_ok=True)
    URL = "http://deeplearning.net/data/mnist/"
    FILENAME = "mnist.pkl.gz"
    if not (PATH / FILENAME).exists():
        content = requests.get(URL + FILENAME).content
        (PATH / FILENAME).open("wb").write(content)
    with gzip.open((PATH / FILENAME).as_posix(), "rb") as f:
        ((x_train, y_train), (x_valid, y_valid), _) = pickle.load(f, encoding="latin-1")
    x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
    x_valid = x_valid.reshape(x_valid.shape[0], x_valid.shape[1], 1)
    y_train = oneHot(y_train)
    y_valid = oneHot(y_valid)
    return x_train, y_train, x_valid, y_valid

