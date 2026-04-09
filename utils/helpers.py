import numpy as np


def one_hot(y, num_classes):
    return np.eye(num_classes, dtype=np.float32)[y]


def create_batches(X, y, batch_size=8, shuffle=True):
    indices = np.arange(len(X))
    if shuffle:
        np.random.shuffle(indices)

    for start in range(0, len(X), batch_size):
        end = start + batch_size
        batch_idx = indices[start:end]
        yield X[batch_idx], y[batch_idx]