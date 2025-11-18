from utils import *
from model.layers import Dense
from utils.data_loader import load_data
from utils.losses import mse, mse_grad
import numpy as np

X_train, y_train, X_test, y_test = load_data()

dense_1 = Dense(X_train[0].shape[0], 100)
dense_2 = Dense(100, 10)

epochs = 100

def evaluate_accuracy(dense_1, dense_2, X, y_true):
    correct = 0
    for i in range(len(X)):
        out = dense_1.forward(X[i])
        out = dense_2.forward(out)
        if np.argmax(out) == np.argmax(y_true[i]):
            correct += 1
    return correct / len(X)

for i in range(epochs):
    p = np.random.permutation(len(X_train))
    X_train = X_train[p]
    y_train = y_train[p]

    for X, y in zip(X_train, y_train):

        out = dense_1.forward(X)
        out = dense_2.forward(out)

        grad = mse_grad(out, y)

        dX = dense_2.backward(grad, 0.01)
        dX = dense_1.backward(dX, 0.01)

    loss = mse(out, y)

    acc = evaluate_accuracy(dense_1, dense_2, X_test[:1000], y_test[:1000])
    print("epoch: ", i, "loss: ", loss, "Accuracy: ", acc)



