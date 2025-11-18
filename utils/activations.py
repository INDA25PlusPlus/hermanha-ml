import numpy as np

def sigmoid(Z: np.ndarray) -> np.ndarray:
    return 1/(1 + np.exp(-Z))

def softmax(Z: np.ndarray) -> np.ndarray:
    # later improvement apperently softmax is the best for classification well for
    # probabilistic classification, as all values adds up to one. Sigmoid would give individual probabilities
    # which would lead to a total probability less than or more than 1, which could be problamatic.

    Z = Z-np.max(Z)
    return np.exp(Z)/np.sum(np.exp(Z))