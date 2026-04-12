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
        self.x_cols = None
        self.output_shape = None

    def _extract_patches(self, x_padded):
        """Return sliding patches with shape (N, H_out, W_out, C, HH, WW)."""
        HH, WW = self.kernel_size, self.kernel_size
        windows = np.lib.stride_tricks.sliding_window_view(
            x_padded,
            window_shape=(HH, WW),
            axis=(2, 3),
        )

        if self.stride > 1:
            windows = windows[:, :, :: self.stride, :: self.stride, :, :]

        # (N, C, H_out, W_out, HH, WW) -> (N, H_out, W_out, C, HH, WW)
        return np.transpose(windows, (0, 2, 3, 1, 4, 5))
        
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
        
        patches = self._extract_patches(self.x_padded)
        self.output_shape = (H_out, W_out)

        # Flatten patches for GEMM: (N*H_out*W_out, C*HH*WW)
        self.x_cols = patches.reshape(N * H_out * W_out, C * HH * WW)
        w_cols = self.Weights.reshape(F, -1)

        out_cols = self.x_cols @ w_cols.T
        out_cols += self.Biases

        out = out_cols.reshape(N, H_out, W_out, F)
        return np.transpose(out, (0, 3, 1, 2)).astype(np.float32, copy=False)
    
    def backward(self, dout):
        """
        dout shape: (N, out_channels, H_out, W_out)
        """
        N, F, H_out, W_out = dout.shape
        _, C, H_padded, W_padded = self.x_padded.shape
        HH, WW = self.kernel_size, self.kernel_size

        dout_cols = np.transpose(dout, (0, 2, 3, 1)).reshape(N * H_out * W_out, F)

        self.db = np.sum(dout_cols, axis=0).astype(np.float32, copy=False)

        dW_cols = dout_cols.T @ self.x_cols
        self.dW = dW_cols.reshape(self.Weights.shape).astype(np.float32, copy=False)

        w_cols = self.Weights.reshape(F, -1)
        dx_cols = dout_cols @ w_cols

        # (N*H_out*W_out, C*HH*WW) -> (N, C, H_out, W_out, HH, WW)
        dx_cols = dx_cols.reshape(N, H_out, W_out, C, HH, WW)
        dx_cols = np.transpose(dx_cols, (0, 3, 1, 2, 4, 5))

        dx_padded = np.zeros_like(self.x_padded, dtype=np.float32)

        # Scatter patches back with only kernel-size loops (typically small).
        for kh in range(HH):
            h_slice = slice(kh, kh + H_out * self.stride, self.stride)
            for kw in range(WW):
                w_slice = slice(kw, kw + W_out * self.stride, self.stride)
                dx_padded[:, :, h_slice, w_slice] += dx_cols[:, :, :, :, kh, kw]
        
        # Unpad dx if padding was added
        if self.padding > 0:
            dx = dx_padded[:, :, self.padding:-self.padding, self.padding:-self.padding]
        else:
            dx = dx_padded
        
        return dx.astype(np.float32, copy=False)

    def update(self, lr):
        self.Weights -= lr * self.dW
        self.Biases -= lr * self.db
    