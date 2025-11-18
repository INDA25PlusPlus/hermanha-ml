import numpy as np

def mse(y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
    return np.square(np.subtract(y_true, y_pred)).mean()

def mse_grad(y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
    return 2 * (y_pred - y_true) / y_pred.size

def rmse(y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
    return np.sqrt(mse(y_pred, y_true))

def cross_entropy(y_pred, y_true) -> np.ndarray:
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1-epsilon)
    return -np.sum(y_true * np.log(y_pred))

def cross_entropy_grad(y_pred, y_true) -> np.ndarray:
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1-epsilon)
    return y_pred - y_true