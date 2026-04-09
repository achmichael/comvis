import numpy as np


class ReLU:
    def __init__(self):
        self.x = None

    def forward(self, x):
        """
        x bisa berbentuk:
        - (N, C, H, W) untuk output conv
        - (N, D) untuk output fully connected
        """
        self.x = x
        return np.maximum(0, x)
    
    def backward(self, dout):
        """
        dout shape sama dengan x
        """
        dx = dout.copy()
        dx[self.x <= 0] = 0
        return dx