import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

# pyrefly: ignore [missing-import]
import matplotlib.pyplot as plt
# pyrefly: ignore [missing-import]
import numpy as np

from dataset.loader import load_dataset
from dataset.preprocess import (
    augment_batch,
    collect_image_paths,
    normalize_image,
    read_image,
    resize_image,
    to_channel_first,
)
from dataset.split import train_val_test_split
from losses.softmax_crossentropy import SoftmaxCrossEntropy
from models.waste_cnn import WasteCNN
from optimizers.sgd import SGD
from utils.helpers import create_batches, one_hot
from utils.metrics import accuracy_from_logits


OUTPUT_DIR = Path("outputs")
LOG_DIR = OUTPUT_DIR / "logs"
WEIGHT_DIR = OUTPUT_DIR / "weights"
PREDICTION_DIR = OUTPUT_DIR / "predictions"


def ensure_output_dirs() -> None:
    for directory in (LOG_DIR, WEIGHT_DIR, PREDICTION_DIR):
        directory.mkdir(parents=True, exist_ok=True)


def _stats(arr: np.ndarray) -> Dict[str, object]:
    return {
        "shape": list(arr.shape),
        "dtype": str(arr.dtype),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "mean": float(np.mean(arr)),
    }


def _to_display_image(arr: np.ndarray) -> Tuple[np.ndarray, str | None]:
    if arr.ndim == 2:
        return arr, "gray"

    if arr.ndim == 3 and arr.shape[2] == 3:
        return arr, None

    if arr.ndim == 3 and arr.shape[0] in (1, 3):
        if arr.shape[0] == 1:
            return arr[0], "gray"
        return np.transpose(arr, (1, 2, 0)), None

    raise ValueError("Array image tidak dikenali untuk ditampilkan")


def trace_preprocessing_steps(
    dataset_root: str,
    img_size: Tuple[int, int] = (32, 32),
    grayscale: bool = False,
    max_images: int = 5,
) -> Tuple[Path, List[Path]]:
    all_images = list(collect_image_paths(dataset_root, recursive=True))
    selected_images = all_images[:max_images]

    if not selected_images:
        raise ValueError("Tidak ada image yang ditemukan untuk debug preprocessing")

    trace_items: List[Dict[str, object]] = []
    preview_paths: List[Path] = []

    print(f"\n== Trace Preprocessing ({len(selected_images)} image pertama) ==")
    for index, image_path in enumerate(selected_images, start=1):
        raw = read_image(image_path, grayscale=grayscale)
        resized = resize_image(raw, img_size=img_size)
        normalized = normalize_image(resized)
        chw = to_channel_first(normalized)

        print(f"[{index}/{len(selected_images)}] {image_path}")
        print(f"  - read_image     : {_stats(raw)}")
        print(f"  - resize_image   : {_stats(resized)}")
        print(f"  - normalize      : {_stats(normalized)}")
        print(f"  - to_channelfirst: {_stats(chw)}")

        trace_items.append(
            {
                "path": str(image_path),
                "read_image": _stats(raw),
                "resize_image": _stats(resized),
                "normalize_image": _stats(normalized),
                "to_channel_first": _stats(chw),
            }
        )

        preview_path = PREDICTION_DIR / f"preprocess_step_{index:02d}.png"
        fig, axes = plt.subplots(1, 4, figsize=(16, 4))

        raw_display, raw_cmap = _to_display_image(raw)
        resized_display, resized_cmap = _to_display_image(resized)
        normalized_display, normalized_cmap = _to_display_image(normalized)
        chw_display, chw_cmap = _to_display_image(chw)

        axes[0].imshow(raw_display.astype(np.uint8), cmap=raw_cmap)
        axes[0].set_title("read_image")
        axes[0].axis("off")

        axes[1].imshow(resized_display.astype(np.uint8), cmap=resized_cmap)
        axes[1].set_title("resize_image")
        axes[1].axis("off")

        axes[2].imshow(np.clip(normalized_display, 0, 1), cmap=normalized_cmap)
        axes[2].set_title("normalize_image")
        axes[2].axis("off")

        axes[3].imshow(np.clip(chw_display, 0, 1), cmap=chw_cmap)
        axes[3].set_title("to_channel_first")
        axes[3].axis("off")

        fig.tight_layout()
        fig.savefig(preview_path, dpi=120)
        plt.close(fig)
        preview_paths.append(preview_path)

    trace_path = LOG_DIR / "preprocessing_trace_first5.json"
    with trace_path.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "img_size": list(img_size),
                "grayscale": grayscale,
                "max_images": max_images,
                "items": trace_items,
            },
            handle,
            indent=2,
        )

    return trace_path, preview_paths


def evaluate(model, criterion, X, y, num_classes, batch_size=32):
    total_loss = 0.0
    total_acc = 0.0
    total_samples = 0

    for xb, yb in create_batches(X, y, batch_size, shuffle=False):
        logits = model.forward(xb)
        yb_onehot = one_hot(yb, num_classes)

        loss = criterion.forward(logits, yb_onehot)
        acc = accuracy_from_logits(logits, yb)

        total_loss += loss * len(xb)
        total_acc += acc * len(xb)
        total_samples += len(xb)

    if total_samples == 0:
        raise ValueError("Dataset evaluasi kosong")

    return total_loss / total_samples, total_acc / total_samples


def train(
    model,
    X_train,
    y_train,
    X_val,
    y_val,
    num_classes,
    epochs=10,
    lr=0.001,
    batch_size=8,
    optimizer=None,
    augment_train: bool = True,
    augmentation_seed: int = 42,
    horizontal_flip_prob: float = 0.5,
    vertical_flip_prob: float = 0.5,
    rotation_range: Tuple[float, float] = (-15.0, 15.0),
    brightness_range: Tuple[float, float] = (0.9, 1.1),
):
    criterion = SoftmaxCrossEntropy()

    if optimizer is None:
        optimizer = SGD(model.layers, lr=lr)
    elif hasattr(optimizer, "set_layers"):
        optimizer.set_layers(model.layers)

    history = {
        "epoch": [],
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
    }

    aug_rng = np.random.default_rng(augmentation_seed) if augment_train else None

    for epoch in range(epochs):
        total_loss = 0.0
        total_acc = 0.0
        total_samples = 0

        for xb, yb in create_batches(X_train, y_train, batch_size=batch_size, shuffle=True):
            if augment_train:
                xb = augment_batch(
                    xb,
                    rng=aug_rng,
                    horizontal_flip_prob=horizontal_flip_prob,
                    vertical_flip_prob=vertical_flip_prob,
                    rotation_range=rotation_range,
                    brightness_range=brightness_range,
                )

            logits = model.forward(xb)

            yb_onehot = one_hot(yb, num_classes)
            loss = criterion.forward(logits, yb_onehot)

            grad = criterion.backward()
            model.backward(grad)
            optimizer.step()

            acc = accuracy_from_logits(logits, yb)

            total_loss += loss * len(xb)
            total_acc += acc * len(xb)
            total_samples += len(xb)

        train_loss = total_loss / total_samples
        train_acc = total_acc / total_samples

        val_loss, val_acc = evaluate(
            model,
            criterion,
            X_val,
            y_val,
            num_classes=num_classes,
            batch_size=batch_size,
        )

        history["epoch"].append(epoch + 1)
        history["train_loss"].append(float(train_loss))
        history["train_acc"].append(float(train_acc))
        history["val_loss"].append(float(val_loss))
        history["val_acc"].append(float(val_acc))

        print(
            f"Epoch {epoch+1:02d} | "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}"
        )

    return history


def save_model_weights(model, file_path: Path) -> Path:
    params = {}
    metadata = {"layers": []}

    for index, layer in enumerate(model.layers):
        layer_meta = {
            "index": index,
            "type": layer.__class__.__name__,
            "params": [],
        }

        for attr in ("Weights", "Biases", "W", "b"):
            if hasattr(layer, attr):
                key = f"layer{index}_{layer.__class__.__name__}_{attr}"
                params[key] = getattr(layer, attr)
                layer_meta["params"].append(attr)

        metadata["layers"].append(layer_meta)

    if not params:
        raise ValueError("Tidak ada parameter model yang bisa disimpan")

    np.savez(file_path, **params)

    meta_path = file_path.with_suffix(".meta.json")
    with meta_path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

    return meta_path


def save_training_history(
    history: Dict[str, List[float]],
    json_path: Path,
    csv_path: Path,
    summary: Dict[str, object],
) -> None:
    payload = {
        "history": history,
        "summary": summary,
    }

    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)

    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["epoch", "train_loss", "train_acc", "val_loss", "val_acc"])

        for i, epoch in enumerate(history["epoch"]):
            writer.writerow(
                [
                    epoch,
                    history["train_loss"][i],
                    history["train_acc"][i],
                    history["val_loss"][i],
                    history["val_acc"][i],
                ]
            )


def plot_training_history(history: Dict[str, List[float]], output_path: Path) -> None:
    epochs = history["epoch"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot(epochs, history["train_loss"], marker="o", label="Train Loss")
    axes[0].plot(epochs, history["val_loss"], marker="o", label="Val Loss")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True, linestyle="--", alpha=0.4)

    axes[1].plot(epochs, history["train_acc"], marker="o", label="Train Acc")
    axes[1].plot(epochs, history["val_acc"], marker="o", label="Val Acc")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()
    axes[1].grid(True, linestyle="--", alpha=0.4)

    fig.tight_layout()
    fig.savefig(output_path, dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    ensure_output_dirs()

    IMG_SIZE = (128, 128)
    GRAYSCALE = False
    EPOCHS = 50
    LR = 0.0003
    BATCH_SIZE = 32

    AUGMENT_TRAIN = True
    AUGMENT_HORIZONTAL_FLIP_PROB = 0.5
    AUGMENT_VERTICAL_FLIP_PROB = 0.5
    AUGMENT_ROTATION_RANGE = (-15.0, 15.0)
    AUGMENT_BRIGHTNESS_RANGE = (0.9, 1.1)
    AUGMENTATION_SEED = 42

    # Batasi hanya 5 image pertama untuk debug proses preprocessing.
    DEBUG_PREPROCESS_LIMIT = 5
    MAX_SAMPLES_PER_CLASS = None

    trace_path, preview_paths = trace_preprocessing_steps(
        "dataset",
        img_size=IMG_SIZE,
        grayscale=GRAYSCALE,
        max_images=DEBUG_PREPROCESS_LIMIT,
    )
    print(f"\nTrace preprocessing tersimpan: {trace_path}")
    print(f"Preview preprocessing tersimpan: {len(preview_paths)} file di {PREDICTION_DIR}")

    print("\nLoading dataset...")
    X, y, class_map = load_dataset(
        "dataset",
        img_size=IMG_SIZE,
        grayscale=GRAYSCALE,
        max_samples_per_class=MAX_SAMPLES_PER_CLASS,
        shuffle=True,
        random_state=42,
        return_class_map=True,
    )

    print(f"Dataset loaded: {len(X)} samples | class_map: {class_map}")

    X_train, y_train, X_val, y_val, X_test, y_test = train_val_test_split(
        X,
        y,
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        stratify=True,
        shuffle=True,
        random_state=42,
    )

    print(f"Split data -> train: {len(X_train)}, val: {len(X_val)}, test: {len(X_test)}")

    if AUGMENT_TRAIN:
        print(
            "Augmentasi train aktif | "
            f"hflip={AUGMENT_HORIZONTAL_FLIP_PROB}, "
            f"vflip={AUGMENT_VERTICAL_FLIP_PROB}, "
            f"rotation={AUGMENT_ROTATION_RANGE}, "
            f"brightness={AUGMENT_BRIGHTNESS_RANGE}"
        )

    num_classes = len(class_map)
    in_channels = 1 if GRAYSCALE else 3
    model = WasteCNN(input_size=IMG_SIZE[0], in_channels=in_channels, num_classes=num_classes)

    history = train(
        model,
        X_train,
        y_train,
        X_val,
        y_val,
        num_classes=num_classes,
        epochs=EPOCHS,
        lr=LR,
        batch_size=BATCH_SIZE,
        augment_train=AUGMENT_TRAIN,
        augmentation_seed=AUGMENTATION_SEED,
        horizontal_flip_prob=AUGMENT_HORIZONTAL_FLIP_PROB,
        vertical_flip_prob=AUGMENT_VERTICAL_FLIP_PROB,
        rotation_range=AUGMENT_ROTATION_RANGE,
        brightness_range=AUGMENT_BRIGHTNESS_RANGE,
    )

    criterion = SoftmaxCrossEntropy()
    test_loss, test_acc = evaluate(
        model,
        criterion,
        X_test,
        y_test,
        num_classes=num_classes,
        batch_size=BATCH_SIZE,
    )
    print(f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    weight_path = WEIGHT_DIR / f"waste_cnn_{timestamp}.npz"
    meta_path = save_model_weights(model, weight_path)
    print(f"Model weights tersimpan: {weight_path}")
    print(f"Model metadata tersimpan: {meta_path}")

    history_json = LOG_DIR / f"train_history_{timestamp}.json"
    history_csv = LOG_DIR / f"train_history_{timestamp}.csv"
    summary = {
        "test_loss": float(test_loss),
        "test_acc": float(test_acc),
        "class_map": class_map,
    }
    save_training_history(history, history_json, history_csv, summary)
    print(f"History JSON tersimpan: {history_json}")
    print(f"History CSV tersimpan: {history_csv}")

    plot_path = LOG_DIR / f"train_plot_{timestamp}.png"
    plot_training_history(history, plot_path)
    print(f"Plot train/val loss & acc tersimpan: {plot_path}")