import numpy as np

class Conv2D:
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        
        scale = np.sqrt(2. / (in_channels * kernel_size * kernel_size))
        # He initialization untuk bobot konvolusi
        self.Weights = np.random.randn(out_channels, in_channels, kernel_size, kernel_size).astype(np.float32) * scale
        self.Biases = np.zeros(out_channels, dtype=np.float32)
        self.dW = np.zeros_like(self.Weights)
        self.db = np.zeros_like(self.Biases)
        
        # cache for backward pass
        self.x = None
        self.x_padded = None
        
    def forward(self, x):
        """
        x shape: (N, C, H, W)
        output shape: (N, out_channels, H_out, W_out)
        """
        self.x = x
        N, C, H, W = x.shape
        F, _, HH, WW = self.Weights.shape
        
        # Calculate output dimensions
        H_out = (H + 2 * self.padding - HH) // self.stride + 1
        W_out = (W + 2 * self.padding - WW) // self.stride + 1
        
        # Pad the input
        if self.padding > 0:
            self.x_padded = np.pad(x, ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)), mode='constant')
        else:
            self.x_padded = x
            
        if H_out <= 0 or W_out <= 0:
            raise ValueError("Output dimensions are negative. Check kernel size, stride, and padding.")
        
        out = np.zeros((N, F, H_out, W_out), dtype=np.float32)
        
        for n in range(N):  # loop over batch
            for f in range(F): # loop over filters
                for h in range(H_out): # loop over output height
                    for w in range(W_out): # loop over output width
                        h_start = h * self.stride
                        h_end = h_start + HH
                        w_start = w * self.stride
                        w_end = w_start + WW
                        
                        out[n, f, h, w] = np.sum(self.x_padded[n, :, h_start:h_end, w_start:w_end] * self.Weights[f]) + self.Biases[f]
        
        return out
    
    def backward(self, dout):
        """
        dout shape: (N, out_channels, H_out, W_out)
        """
        N, F, H_out, W_out = dout.shape
        _, C, H_padded, W_padded = self.x_padded.shape
        HH, WW = self.kernel_size, self.kernel_size
        
        self.dW = np.zeros_like(self.Weights)
        self.db = np.zeros_like(self.Biases)
        dx_padded = np.zeros_like(self.x_padded)
        
        for n in range(N):
            for f in range(F):
                for h in range(H_out):
                    for w in range(W_out):
                        h_start = h * self.stride
                        h_end = h_start + HH
                        w_start = w * self.stride
                        w_end = w_start + WW
                        
                        self.dW[f] += self.x_padded[n, :, h_start:h_end, w_start:w_end] * dout[n, f, h, w]
                        self.db[f] += dout[n, f, h, w]
                        dx_padded[n, :, h_start:h_end, w_start:w_end] += self.Weights[f] * dout[n, f, h, w]
        
        # Unpad dx if padding was added
        if self.padding > 0:
            dx = dx_padded[:, :, self.padding:-self.padding, self.padding:-self.padding]
        else:
            dx = dx_padded
        
        return dx

    def update(self, lr):
        self.Weights -= lr * self.dW
        self.Biases -= lr * self.db
    