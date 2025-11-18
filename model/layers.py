import numpy as np
from utils.activations import softmax, sigmoid

class Dense:
    def __init__(self, input_dim: int, output_dim: int, activation: str = "softmax"):
        self.W = 0.01 * np.random.randn(input_dim, output_dim)
        self.b = np.zeros(output_dim)
        self.activation = activation
    
    def forward(self, X) -> np.ndarray:
        wb = X @ self.W + self.b
        if self.activation == "sigmoid":
            return(sigmoid(wb))
        elif self.activation == "softmax":
            return(softmax(wb))
        else:
            raise Exception("activation function not allowed")
        
    
            
        
    