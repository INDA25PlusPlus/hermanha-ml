from utils import *
from model.layers import Dense
from utils.data_loader import load_data

X_train, y_train, X_test, y_test = load_data()

for i in range(1):

    dense_1 = Dense(X_train[i].shape[0], 100)
    dense_2 = Dense(100, 10)

    y = dense_1.forward(X_train[i])
    y = dense_2.forward(y)

    print(y)
    