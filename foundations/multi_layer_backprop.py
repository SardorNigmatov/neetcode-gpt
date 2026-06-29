import numpy as np
from typing import List


class Solution:
    def forward_and_backward(self,
                              x: List[float],
                              W1: List[List[float]], b1: List[float],
                              
                              W2: List[List[float]], b2: List[float],
                              y_true: List[float]) -> dict:
        # Architecture: x -> Linear(W1, b1) -> ReLU -> Linear(W2, b2) -> predictions
        # Loss: MSE = mean((predictions - y_true)^2)
        
        # Convert inputs to numpy arrays
        x = np.array(x)
        W1 = np.array(W1)
        b1 = np.array(b1)
        W2 = np.array(W2)
        b2 = np.array(b2)
        y_true = np.array(y_true)

        # Forward pass (using Transpose to match input_size x hidden_size logic)
        z1 = W1 @ x + b1
        a1 = np.maximum(0, z1)   # ReLU

        predictions = W2 @ a1 + b2

        loss = np.mean((predictions - y_true) ** 2)

        # dLoss / dPredictions
        n = y_true.size
        d_pred = 2 * (predictions - y_true) / n

        # Layer 2 gradients
        dW2 = np.outer(d_pred, a1)
        db2 = d_pred

        # Backprop to hidden layer
        da1 = W2.T @ d_pred

        # ReLU gradient
        dz1 = da1 * (z1 > 0)

        # Layer 1 gradients
        dW1 = np.outer(dz1, x)
        db1 = dz1

        return {
            "loss": round(float(loss), 4),
            "dW1": np.round(dW1, 4).tolist(),
            "db1": np.round(db1, 4).tolist(),
            "dW2": np.round(dW2, 4).tolist(),
            "db2": np.round(db2, 4).tolist()
        }
