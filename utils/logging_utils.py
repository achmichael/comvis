import numpy as np
import logging
import json
from pathlib import Path
from datetime import datetime

logger = logging.getLogger("cnn_pipeline")


def setup_pipeline_logger(log_dir: str = "outputs/logs", level=logging.DEBUG) -> logging.Logger:
    """Setup logger yang menulis ke file dan console."""
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = Path(log_dir) / f"pipeline_trace_{timestamp}.log"

    formatter = logging.Formatter(
        "[%(asctime)s] %(levelname)s - %(message)s",
        datefmt="%H:%M:%S",
    )

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)

    logger.setLevel(level)
    # Hindari duplikasi handler
    logger.handlers.clear()
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    logger.info(f"Pipeline logger aktif. Log file: {log_file}")
    return logger


def tensor_summary(name: str, arr: np.ndarray, show_sample: bool = True, sample_size: int = 5) -> str:
    """Buat ringkasan tensor: shape, dtype, min, max, mean, std, dan sample nilai."""
    lines = [
        f"  [{name}]",
        f"    Shape : {arr.shape}",
        f"    Dtype : {arr.dtype}",
        f"    Min   : {float(np.min(arr)):.6f}",
        f"    Max   : {float(np.max(arr)):.6f}",
        f"    Mean  : {float(np.mean(arr)):.6f}",
        f"    Std   : {float(np.std(arr)):.6f}",
    ]
    if show_sample:
        flat = arr.flatten()
        n = min(sample_size, len(flat))
        sample = flat[:n]
        lines.append(f"    Sample (first {n}): {np.array2string(sample, precision=4, separator=', ')}")
    return "\n".join(lines)


def log_tensor(name: str, arr: np.ndarray, show_sample: bool = True) -> None:
    """Log ringkasan tensor ke logger."""
    logger.debug(tensor_summary(name, arr, show_sample))
