import numpy as np
from layers.conv2d import Conv2D

# batch 2 gambar grayscale ukuran 5x5
x = np.random.rand(8, 3, 32, 32).astype(np.float32)

conv = Conv2D(
    in_channels=3,
    out_channels=4,
    kernel_size=3,
    stride=1,
    padding=0
)

out = conv.forward(x)

print("Input shape :", x.shape)
print("Weight shape:", conv.Weights.shape)
print("Bias shape  :", conv.Biases.shape)
print("Output shape:", out.shape)
print("Output:\n", out)