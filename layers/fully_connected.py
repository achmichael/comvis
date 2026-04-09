import numpy as np

class FullyConnected:
    def __init__(self, in_features, out_features):
        self.in_features = in_features
        self.out_features = out_features
        scale = np.sqrt(2.0 / in_features)
        self.W = (np.random.randn(out_features, in_features) * scale).astype(np.float32)
        self.b = np.zeros(out_features, dtype=np.float32)
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)

        # cache untuk backward nanti
        self.x = None
    
    def forward(self, x):
        """
        x shape: (N, in_features)
        output shape: (N, out_features)
        """
        if x.ndim != 2:
            raise ValueError("Input Dense harus memiliki shape (N, in_features)")

        N, D = x.shape

        if D != self.in_features:
            raise ValueError(
                f"Jumlah fitur input tidak cocok. Expected {self.in_features}, dapat {D}"
            )

        self.x = x

        out = x @ self.W.T + self.b
        return out
    
    def backward(self, dout):
        """
        dout shape: (N, out_features)
        """
        N, D = self.x.shape

        # Gradien untuk bobot dan biasE
        self.dW = dout.T @ self.x
        self.db = np.sum(dout, axis=0)

        # Gradien untuk input (jika diperlukan untuk backprop lebih lanjut)
        dx = dout @ self.W

        return dx

    def update(self, lr):
        self.W -= lr * self.dW
        self.b -= lr * self.db