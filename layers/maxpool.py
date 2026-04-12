import numpy as np


class MaxPool2D:
    def __init__(self, pool_size=1, stride=1):
        """pool_size: ukuran jendela pooling (misal 2 untuk 2x2)
           stride: langkah pergeseran jendela pooling
        """
        self.pool_size = pool_size
        self.stride = stride
        self.x = None  # cache untuk backward pass
        self.max_mask = None
        
        
        
    def forward(self, x):
        """
        x shape: (N, C, H, W)
        output shape: (N, C, H_out, W_out)
        """
        if x.ndim != 4:
            raise ValueError("Input harus memiliki shape (N, C, H, W)")

        self.x = x

        N, C, H, W = x.shape
        P = self.pool_size
        S = self.stride

        H_out = (H - P) // S + 1
        W_out = (W - P) // S + 1

        if H_out <= 0 or W_out <= 0:
            raise ValueError("Ukuran output <= 0. Cek pool_size dan stride.")

        windows = np.lib.stride_tricks.sliding_window_view(
            x,
            window_shape=(P, P),
            axis=(2, 3),
        )

        if S > 1:
            windows = windows[:, :, ::S, ::S, :, :]

        max_vals = np.max(windows, axis=(-2, -1), keepdims=True)
        self.max_mask = windows == max_vals

        out = np.squeeze(max_vals, axis=(-2, -1))
        return out.astype(np.float32, copy=False)
    
    def backward(self, dout):
        """
        dout shape: (N, C, H_out, W_out)
        output shape: (N, C, H, W)
        """
        N, C, H, W = self.x.shape
        P = self.pool_size
        S = self.stride
        H_out = (H - P) // S + 1
        W_out = (W - P) // S + 1

        if self.max_mask is None:
            raise ValueError("Forward harus dipanggil sebelum backward")

        dx = np.zeros_like(self.x, dtype=np.float32)

        # Distribusikan gradien ke semua elemen max (sesuai perilaku implementasi awal).
        for kh in range(P):
            h_slice = slice(kh, kh + H_out * S, S)
            for kw in range(P):
                w_slice = slice(kw, kw + W_out * S, S)
                dx[:, :, h_slice, w_slice] += dout * self.max_mask[:, :, :, :, kh, kw]

        return dx.astype(np.float32, copy=False)
