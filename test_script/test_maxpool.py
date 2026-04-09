import numpy as np
from layers.conv2d import Conv2D
from layers.maxpool import MaxPool2D

x = np.random.rand(8, 3, 32, 32).astype(np.float32)

conv = Conv2D(in_channels=3, out_channels=4, kernel_size=3, stride=1, padding=0)
pool = MaxPool2D(pool_size=2, stride=2)

out_conv = conv.forward(x)
out_pool = pool.forward(out_conv)

print("Input     :", x.shape)
print("After Conv:", out_conv.shape)
print("After Pool:", out_pool.shape)