import numpy as np

def mse(y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
    return np.square(np.subtract(y_true, y_pred)).mean()

def mse_grad(y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
    return 2 * (y_pred - y_true) / y_pred.size

def rmse(y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
    return np.sqrt(mse(y_pred, y_true))

def cross_entropy(y_pred, y_true) -> np.ndarray:
    return -np.mean(np.sum(y_true @ np.log(y_pred), axis=1))