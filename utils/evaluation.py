import numpy as np
from model.layers import Dense

def evaluate_accuracy(dense_1: Dense, dense_2: Dense, X, y_true) -> float:
    correct = 0
    for i in range(len(X)):
        out = dense_1.forward(X[i])
        out = dense_2.forward(out)
        if np.argmax(out) == np.argmax(y_true[i]):
            correct += 1
    return correct / len(X)