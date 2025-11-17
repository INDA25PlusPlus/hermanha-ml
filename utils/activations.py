import numpy as np

def sigmoid(Z: np.ndarray) -> np.ndarray:
    return 1/(1 + np.exp(-Z))

def softmax(Z: np.ndarray) -> np.ndarray:
    # later improvement
    pass