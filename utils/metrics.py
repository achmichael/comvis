import numpy as np


def accuracy_from_logits(logits, y_true):
    preds = np.argmax(logits, axis=1)
    return np.mean(preds == y_true)