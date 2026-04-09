from layers.conv2d import Conv2D
from layers.maxpool import MaxPool2D
from layers.flatten import Flatten
from layers.fully_connected import FullyConnected
from activations.relu import ReLU
import numpy as np


class WasteCNN:
    def __init__(self, input_size=32, in_channels=3, num_classes=2):
        self.conv1 = Conv2D(in_channels=in_channels, out_channels=4, kernel_size=3, stride=1, padding=0)
        self.relu1 = ReLU()
        self.pool1 = MaxPool2D(pool_size=2, stride=2)
        self.flatten = Flatten()

        # hitung otomatis dimensi flatten
        dummy = np.zeros((1, in_channels, input_size, input_size), dtype=np.float32)
        out = self.conv1.forward(dummy)
        out = self.relu1.forward(out)
        out = self.pool1.forward(out)
        flat_dim = out.reshape(1, -1).shape[1]

        self.dense1 = FullyConnected(in_features=flat_dim, out_features=64)
        self.relu2 = ReLU()
        self.dense2 = FullyConnected(in_features=64, out_features=num_classes)

        self.layers = [
            self.conv1,
            self.relu1,
            self.pool1,
            self.flatten,
            self.dense1,
            self.relu2,
            self.dense2,
        ]

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, grad):
        for layer in reversed(self.layers):
            grad = layer.backward(grad)

    def update(self, lr):
        for layer in self.layers:
            if hasattr(layer, "update"):
                layer.update(lr)