import numpy as np

class Flatten:
    def __init__(self):
        self.input_shape = None  # cache untuk backward pass
    
    def forward(self, x):
        """
        x shape: (N, C, H, W)
        output shape: (N, C*H*W)
        """
        if x.ndim != 4:
            raise ValueError("Input harus memiliki shape (N, C, H, W)")

        self.input_shape = x.shape
        N = x.shape[0]
        out = x.reshape(N, -1)
        return out
    
    def backward(self, dout):
        """
        dout shape: (N, C*H*W)
        output shape: (N, C, H, W)
        """
        if self.input_shape is None:
            raise ValueError("Tidak ada input yang disimpan untuk backward pass")

        return dout.reshape(self.input_shape)