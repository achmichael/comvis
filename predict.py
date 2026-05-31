import argparse
import logging
from pathlib import Path
from typing import List, Sequence

import numpy as np

from dataset.preprocess import preprocess_image
from models.waste_cnn import WasteCNN
from utils.logging_utils import setup_pipeline_logger

logger = logging.getLogger("cnn_pipeline")


def _parse_class_names(raw: str) -> List[str]:
	class_names = [item.strip() for item in raw.split(",") if item.strip()]
	if len(class_names) < 2:
		raise ValueError("class_names minimal harus berisi 2 kelas")
	return class_names


def _softmax(logits: np.ndarray) -> np.ndarray:
	shifted = logits - np.max(logits, axis=1, keepdims=True)
	exp_scores = np.exp(shifted)
	return exp_scores / np.sum(exp_scores, axis=1, keepdims=True)


def load_model_weights(model: WasteCNN, model_path: Path) -> None:
	if not model_path.exists():
		raise FileNotFoundError(f"File model tidak ditemukan: {model_path}")

	with np.load(model_path) as params:
		missing_keys: List[str] = []

		for index, layer in enumerate(model.layers):
			for attr in ("Weights", "Biases", "W", "b"):
				if not hasattr(layer, attr):
					continue

				key = f"layer{index}_{layer.__class__.__name__}_{attr}"
				if key not in params:
					missing_keys.append(key)
					continue

				expected_shape = getattr(layer, attr).shape
				loaded_array = params[key]

				if loaded_array.shape != expected_shape:
					raise ValueError(
						"Shape parameter tidak cocok untuk "
						f"{key}. Expected {expected_shape}, dapat {loaded_array.shape}"
					)

				setattr(layer, attr, loaded_array.astype(np.float32))

		if missing_keys:
			missing_preview = ", ".join(missing_keys[:5])
			raise KeyError(
				"Ada parameter model yang tidak ditemukan di file bobot: "
				f"{missing_preview}"
			)


def predict_image(
	model: WasteCNN,
	image_path: Path,
	class_names: Sequence[str],
	img_size: int,
	grayscale: bool,
	verbose_log: bool = False,
) -> tuple[str, float, np.ndarray]:
	if not image_path.exists():
		raise FileNotFoundError(f"File image tidak ditemukan: {image_path}")

	image_chw = preprocess_image(
		image_path,
		img_size=(img_size, img_size),
		grayscale=grayscale,
		normalize=True,
	)

	x = np.expand_dims(image_chw, axis=0).astype(np.float32)
	logits = model.forward(x, verbose_log=verbose_log)
	probs = _softmax(logits)[0]

	pred_idx = int(np.argmax(probs))
	pred_label = class_names[pred_idx]
	confidence = float(probs[pred_idx])

	if verbose_log:
		logger.info(f"Final Prediction: {pred_label} (confidence: {confidence:.4f})")
		for idx, cls in enumerate(class_names):
			logger.info(f"  {cls}: {float(probs[idx]):.4f}")

	return pred_label, confidence, probs


def build_arg_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(
		description="Prediksi gambar sampah: organic atau inorganic"
	)
	parser.add_argument(
		"--model_path",
		required=True,
		help="Path ke file bobot model (.npz)",
	)
	parser.add_argument(
		"--image_path",
		required=True,
		help="Path ke file gambar yang ingin diprediksi",
	)
	parser.add_argument(
		"--img_size",
		type=int,
		default=32,
		help="Ukuran gambar (height = width). Default: 32",
	)
	parser.add_argument(
		"--grayscale",
		action="store_true",
		help="Gunakan preprocessing grayscale (1 channel)",
	)
	parser.add_argument(
		"--class_names",
		default="inorganic,organic",
		help="Daftar nama kelas dipisah koma sesuai indeks model",
	)
	parser.add_argument(
		"--verbose_log",
		action="store_true",
		help="Aktifkan logging detail proses forward pass CNN",
	)
	return parser


def main() -> None:
	parser = build_arg_parser()
	args = parser.parse_args()

	class_names = _parse_class_names(args.class_names)
	in_channels = 1 if args.grayscale else 3
	model = WasteCNN(
		input_size=args.img_size,
		in_channels=in_channels,
		num_classes=len(class_names),
	)

	model_path = Path(args.model_path)
	image_path = Path(args.image_path)

	if args.verbose_log:
		setup_pipeline_logger()

	load_model_weights(model, model_path)
	pred_label, confidence, probs = predict_image(
		model,
		image_path,
		class_names=class_names,
		img_size=args.img_size,
		grayscale=args.grayscale,
		verbose_log=args.verbose_log,
	)

	print(f"Image path  : {image_path}")
	print(f"Model path  : {model_path}")
	print(f"Prediction  : {pred_label}")
	print(f"Confidence  : {confidence:.4f}")
	print("Class scores:")
	for idx, class_name in enumerate(class_names):
		print(f"  - {class_name:10s}: {float(probs[idx]):.4f}")


if __name__ == "__main__":
	main()
