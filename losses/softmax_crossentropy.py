import numpy as np


class SoftmaxCrossEntropy:
    def __init__(self):
        self.probs = None
        self.y_true = None

    def forward(self, logits, y_true):
        """
        logits: (N, num_classes)
        y_true: (N, num_classes) one-hot
        """
        shifted = logits - np.max(logits, axis=1, keepdims=True)
        exp_scores = np.exp(shifted)
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

        self.probs = probs
        self.y_true = y_true

        loss = -np.sum(y_true * np.log(probs + 1e-12)) / logits.shape[0]
        return loss

    def backward(self):
        """
        grad loss terhadap logits
        """
        N = self.y_true.shape[0]
        return (self.probs - self.y_true) / N