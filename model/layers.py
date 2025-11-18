import numpy as np
from utils.activations import softmax, sigmoid

class Dense:
    def __init__(self, input_dim: int, output_dim: int, activation: str = "sigmoid"):
        self.W = 0.01 * np.random.randn(input_dim, output_dim)
        self.b = np.zeros(output_dim)
        self.activation = activation
        self.X = None
        self.Z = None
    
    def forward(self, X) -> np.ndarray:
        self.X = X
        self.Z = X @ self.W + self.b
        if self.activation == "sigmoid":
            self.Y = sigmoid(self.Z)
        elif self.activation == "softmax":
            self.Y = softmax(self.Z)
        else:
            raise Exception("activation function not allowed")
        return self.Y
    
    def backward(self, dA, learning_rate):
        if self.activation == "sigmoid":
            dZ = dA * self.Y * (1-self.Y)
        elif self.activation == "softmax":
            # TODO
            pass
        else: 
            raise Exception("unsupported activation")
        
        X = self.X.reshape(1, -1)
        dZ = dZ.reshape(1, -1)
        
        dW = X.T @ dZ
        dB = np.sum(dZ, axis=0)
        dX = dZ @ self.W.T

        self.W -= learning_rate * dW
        self.b -= learning_rate * dB

        return dX
    
    def save(self, path):
        np.savez(path, W=self.W, b=self.b)

    def load(self, path):
        data = np.load(path)
        self.W = data["W"]
        self.b = data["b"]
        data.close()
