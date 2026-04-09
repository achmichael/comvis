import numpy as np

class MaxPool2D:
    def __init__(self, pool_size=1, stride=1):
        """pool_size: ukuran jendela pooling (misal 2 untuk 2x2)
           stride: langkah pergeseran jendela pooling
        """
        self.pool_size = pool_size
        self.stride = stride
        self.x = None  # cache untuk backward pass
        
        
        
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

        out = np.zeros((N, C, H_out, W_out), dtype=np.float32)

        for n in range(N):        # batch
            for c in range(C):    # channel
                for i in range(H_out):
                    for j in range(W_out):
                        h_start = i * S
                        h_end = h_start + P
                        w_start = j * S
                        w_end = w_start + P

                        region = x[n, c, h_start:h_end, w_start:w_end]
                        out[n, c, i, j] = np.max(region)

        return out
    
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

        dx = np.zeros_like(self.x)

        for n in range(N):
            for c in range(C):
                for i in range(H_out):
                    for j in range(W_out):
                        h_start = i * S
                        h_end = h_start + P
                        w_start = j * S
                        w_end = w_start + P

                        region = self.x[n, c, h_start:h_end, w_start:w_end]
                        max_val = np.max(region)

                        # Set gradien hanya untuk elemen yang merupakan max
                        mask = (region == max_val)
                        dx[n, c, h_start:h_end, w_start:w_end] += mask * dout[n, c, i, j]

        return dx
