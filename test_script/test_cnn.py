import numpy as np
from layers.conv2d import Conv2D
from layers.maxpool import MaxPool2D
from layers.flatten import Flatten
from layers.fully_connected import FullyConnected
from activations.relu import ReLU

x = np.random.rand(8, 3, 32, 32).astype(np.float32)

conv = Conv2D(in_channels=3, out_channels=4, kernel_size=3, stride=1, padding=0)
relu1 = ReLU()
pool = MaxPool2D(pool_size=2, stride=2)
flatten = Flatten()
dense1 = FullyConnected(in_features=4 * 15 * 15, out_features=64)
relu2 = ReLU()

out = conv.forward(x)
print("After Conv   :", out.shape)

out = relu1.forward(out)
print("After ReLU 1 :", out.shape)

out = pool.forward(out)
print("After Pool   :", out.shape)

out = flatten.forward(out)
print("After Flatten:", out.shape)

out = dense1.forward(out)
print("After Dense1 :", out.shape)

out = relu2.forward(out)
print("After ReLU 2 :", out.shape)