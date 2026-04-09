import numpy as np


class SGD:
	"""Optimizer Stochastic Gradient Descent untuk layer custom NumPy."""

	def __init__(self, layers=None, lr=1e-3, momentum=0.0, weight_decay=0.0):
		if lr <= 0:
			raise ValueError("lr harus > 0")
		if not (0.0 <= momentum < 1.0):
			raise ValueError("momentum harus di rentang [0, 1)")
		if weight_decay < 0:
			raise ValueError("weight_decay tidak boleh negatif")

		self.layers = list(layers) if layers is not None else []
		self.lr = float(lr)
		self.momentum = float(momentum)
		self.weight_decay = float(weight_decay)
		self._velocity = {}

	def set_layers(self, layers):
		self.layers = list(layers)

	def _iter_params(self, layer):
		# Konvensi parameter/gradien yang dipakai di proyek ini.
		param_map = (
			("Weights", "dW"),
			("Biases", "db"),
			("W", "dW"),
			("b", "db"),
		)
		for param_attr, grad_attr in param_map:
			if hasattr(layer, param_attr) and hasattr(layer, grad_attr):
				yield param_attr, grad_attr

	def step(self, layers=None):
		target_layers = self.layers if layers is None else list(layers)

		for layer in target_layers:
			found_param = False

			for param_attr, grad_attr in self._iter_params(layer):
				found_param = True

				param = getattr(layer, param_attr)
				grad = getattr(layer, grad_attr)

				if grad is None:
					continue

				if param.shape != grad.shape:
					raise ValueError(
						f"Shape grad tidak cocok untuk {layer.__class__.__name__}.{param_attr}: "
						f"{grad.shape} vs {param.shape}"
					)

				grad_update = grad
				if self.weight_decay > 0.0:
					grad_update = grad_update + self.weight_decay * param

				if self.momentum > 0.0:
					velocity_key = (id(layer), param_attr)
					if velocity_key not in self._velocity:
						self._velocity[velocity_key] = np.zeros_like(param)

					v = self._velocity[velocity_key]
					v *= self.momentum
					v += grad_update
					grad_update = v

				param -= self.lr * grad_update

			# Fallback jika layer memakai API update(lr) kustom tanpa akses grad langsung.
			if not found_param and hasattr(layer, "update"):
				layer.update(self.lr)

