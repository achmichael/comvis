import numpy as np
from layers.fully_connected import FullyConnected
from activations.relu import ReLU

x = np.random.randn(8, 900).astype(np.float32)

dense = FullyConnected(in_features=900, out_features=64)
relu = ReLU()

out_dense = dense.forward(x)
out_relu = relu.forward(out_dense)

print("After Dense:", out_dense.shape)
print("After ReLU :", out_relu.shape)