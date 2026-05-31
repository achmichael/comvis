import copy
import logging

from layers.conv2d import Conv2D
from layers.maxpool import MaxPool2D
from layers.flatten import Flatten
from layers.fully_connected import FullyConnected
from activations.relu import ReLU
from utils.logging_utils import log_tensor
import numpy as np

logger = logging.getLogger("cnn_pipeline")


class WasteCNN:
    def __init__(self, input_size=32, in_channels=3, num_classes=2):
        self.conv1 = Conv2D(in_channels=in_channels, out_channels=4, kernel_size=3, stride=1, padding=0)
        self.relu1 = ReLU()
        self.pool1 = MaxPool2D(pool_size=2, stride=2)
        self.flatten = Flatten()

        dummy = np.zeros((1, in_channels, input_size, input_size), dtype=np.float32)
        
        out = self.conv1.forward(dummy)
        print("output convolution layer" + str(out.shape))
        out = self.relu1.forward(out)
        print("output relu layer" + str(out.shape))
        out = self.pool1.forward(out)
        print("output pool layer" + str(out.shape))
        flat_dim = out.reshape(1, -1).shape[1]
        print("flat_dim", flat_dim)

        self.dense1 = FullyConnected(in_features=flat_dim, out_features=64)
        self.relu2 = ReLU()
        self.dense2 = FullyConnected(in_features=64, out_features=num_classes)
        self.relu3 = ReLU()

        self.layers = [
            self.conv1,
            self.relu1,
            self.pool1,
            self.flatten,
            self.dense1,
            self.relu2,
            self.dense2,
            self.relu3,
        ]

    def forward(self, x, verbose_log: bool = False):
        if verbose_log:
            self._forward_with_logging(x)
            # tetap jalankan forward biasa untuk return value
            # tapi kita sudah log di _forward_with_logging
            # jadi pakai cached result
            return self._last_logged_output

        for layer in self.layers:
            x = layer.forward(x)
        return x

    def _forward_with_logging(self, x):
        """Forward pass dengan logging detail setiap tahap."""
        separator = "=" * 70

        logger.info(f"\n{separator}")
        logger.info("FORWARD PASS - PIPELINE LOGGING")
        logger.info(f"{separator}")

        # ── 1. Input gambar (pixel) ──
        logger.info("\n[STEP 1] INPUT IMAGE (Pixel Values)")
        log_tensor("Input Image", x)
        logger.info(f"  Jumlah gambar dalam batch: {x.shape[0]}")
        logger.info(f"  Channel: {x.shape[1]}, Height: {x.shape[2]}, Width: {x.shape[3]}")

        # Tampilkan SELURUH pixel per sample, per channel
        old_opts = np.get_printoptions()
        np.set_printoptions(threshold=np.inf, linewidth=200, precision=4, suppress=True)
        channel_names = {1: ["Gray"], 3: ["R", "G", "B"]}
        ch_labels = channel_names.get(x.shape[1], [f"Ch{i}" for i in range(x.shape[1])])
        for sample_idx in range(x.shape[0]):
            logger.info(f"\n  ---- Sample-{sample_idx} ALL PIXELS ----")
            for ch_idx in range(x.shape[1]):
                pixel_matrix = x[sample_idx, ch_idx]  # (H, W)
                logger.info(f"  Channel {ch_labels[ch_idx]} (index={ch_idx}), "
                           f"size {pixel_matrix.shape[0]}x{pixel_matrix.shape[1]}:")
                logger.info(f"\n{np.array2string(pixel_matrix, separator=', ')}")
        np.set_printoptions(**old_opts)

        # ── 2. Kernel / Filter Conv Layer ──
        logger.info(f"\n[STEP 2] KERNEL / FILTER (Conv2D Weights)")
        log_tensor("Conv1 Weights (Kernel)", self.conv1.Weights)
        log_tensor("Conv1 Biases", self.conv1.Biases)
        logger.info(f"  Jumlah filter: {self.conv1.out_channels}")
        logger.info(f"  Kernel size: {self.conv1.kernel_size}x{self.conv1.kernel_size}")
        logger.info(f"  Stride: {self.conv1.stride}, Padding: {self.conv1.padding}")
        # Log setiap kernel individual
        for f_idx in range(self.conv1.Weights.shape[0]):
            logger.debug(f"  Kernel filter-{f_idx}:")
            for c_idx in range(self.conv1.Weights.shape[1]):
                kernel_2d = self.conv1.Weights[f_idx, c_idx]
                logger.debug(f"    Channel-{c_idx}:\n{kernel_2d}")

        # ── 3. Feature Map (output konvolusi mentah) ──
        conv_out = self.conv1.forward(x)
        logger.info(f"\n[STEP 3] FEATURE MAP (Output Konvolusi)")
        log_tensor("Feature Map (Conv1 output)", conv_out)
        logger.info(f"  Jumlah feature map: {conv_out.shape[1]}")
        logger.info(f"  Ukuran tiap feature map: {conv_out.shape[2]}x{conv_out.shape[3]}")
        # Log sample feature map pertama dari batch pertama
        if conv_out.shape[0] > 0:
            for fm_idx in range(min(conv_out.shape[1], 4)):
                fm = conv_out[0, fm_idx]
                logger.debug(f"  Feature Map [{fm_idx}] (batch-0) - "
                           f"min: {fm.min():.4f}, max: {fm.max():.4f}, mean: {fm.mean():.4f}")

        # ── 4. Output Conv Layer (sama dengan feature map di arsitektur ini) ──
        logger.info(f"\n[STEP 4] OUTPUT CONV LAYER")
        logger.info(f"  Output conv layer = Feature map dari step 3")
        log_tensor("Conv Layer Output", conv_out)

        # ── 5. ReLU ──
        relu_out = self.relu1.forward(conv_out)
        logger.info(f"\n[STEP 5] ReLU ACTIVATION")
        log_tensor("ReLU Output", relu_out)
        num_zero = int(np.sum(relu_out == 0))
        total = relu_out.size
        logger.info(f"  Nilai yang di-nol-kan (negatif -> 0): {num_zero}/{total} "
                   f"({num_zero/total*100:.1f}%)")
        logger.info(f"  Nilai yang lolos (positif): {total - num_zero}/{total} "
                   f"({(total-num_zero)/total*100:.1f}%)")

        # ── 6. MaxPool ──
        pool_out = self.pool1.forward(relu_out)
        logger.info(f"\n[STEP 6] MAX POOLING")
        log_tensor("MaxPool Output", pool_out)
        logger.info(f"  Pool size: {self.pool1.pool_size}x{self.pool1.pool_size}")
        logger.info(f"  Stride: {self.pool1.stride}")
        logger.info(f"  Ukuran sebelum pool: {relu_out.shape[2]}x{relu_out.shape[3]}")
        logger.info(f"  Ukuran sesudah pool: {pool_out.shape[2]}x{pool_out.shape[3]}")
        logger.info(f"  Reduksi dimensi: {relu_out.shape[2]}x{relu_out.shape[3]} -> "
                   f"{pool_out.shape[2]}x{pool_out.shape[3]}")

        # ── 7. Flatten ──
        flat_out = self.flatten.forward(pool_out)
        logger.info(f"\n[STEP 7] FLATTEN")
        log_tensor("Flatten Output", flat_out)
        logger.info(f"  Sebelum flatten: {pool_out.shape}")
        logger.info(f"  Sesudah flatten: {flat_out.shape}")
        logger.info(f"  Total fitur per sample: {flat_out.shape[1]}")

        # ── 8. Fully Connected Layers ──
        logger.info(f"\n[STEP 8] FULLY CONNECTED LAYERS")

        # Dense1
        dense1_out = self.dense1.forward(flat_out)
        logger.info(f"  -- Dense1 (FC1) --")
        log_tensor("Dense1 Weights", self.dense1.W)
        log_tensor("Dense1 Biases", self.dense1.b)
        log_tensor("Dense1 Output", dense1_out)
        logger.info(f"  Input features: {self.dense1.in_features}")
        logger.info(f"  Output features: {self.dense1.out_features}")

        # ReLU2
        relu2_out = self.relu2.forward(dense1_out)
        logger.info(f"  -- ReLU2 (setelah Dense1) --")
        log_tensor("ReLU2 Output", relu2_out)
        num_zero2 = int(np.sum(relu2_out == 0))
        total2 = relu2_out.size
        logger.info(f"  Nilai yang di-nol-kan: {num_zero2}/{total2} ({num_zero2/total2*100:.1f}%)")

        # Dense2 (output layer)
        dense2_out = self.dense2.forward(relu2_out)
        logger.info(f"  -- Dense2 (FC2 - Output Layer) --")
        log_tensor("Dense2 Weights", self.dense2.W)
        log_tensor("Dense2 Biases", self.dense2.b)
        log_tensor("Dense2 Output (Logits)", dense2_out)
        logger.info(f"  Input features: {self.dense2.in_features}")
        logger.info(f"  Output features (num_classes): {self.dense2.out_features}")

        # ── 9. Hasil Prediksi ──
        logger.info(f"\n[STEP 9] HASIL PREDIKSI")
        log_tensor("Raw Logits", dense2_out)

        # Hitung softmax untuk logging
        shifted = dense2_out - np.max(dense2_out, axis=1, keepdims=True)
        exp_scores = np.exp(shifted)
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        log_tensor("Softmax Probabilities", probs)

        pred_indices = np.argmax(probs, axis=1)
        pred_confidences = np.max(probs, axis=1)
        logger.info(f"  Predicted class indices: {pred_indices}")
        logger.info(f"  Predicted confidences: {np.array2string(pred_confidences, precision=4)}")

        for i in range(min(len(pred_indices), 5)):  # log max 5 sample
            logger.info(f"  Sample-{i}: class={pred_indices[i]}, "
                       f"confidence={pred_confidences[i]:.4f}, "
                       f"probs={np.array2string(probs[i], precision=4)}")

        logger.info(f"\n{separator}")
        logger.info("FORWARD PASS SELESAI")
        logger.info(f"{separator}\n")

        self._last_logged_output = dense2_out

    def backward(self, grad):
        for layer in reversed(self.layers):
            grad = layer.backward(grad)

    def update(self, lr):
        for layer in self.layers:
            if hasattr(layer, "update"):
                layer.update(lr)

    def get_weights_copy(self):
        """Kembalikan deep copy semua parameter (weights & biases) tiap layer."""
        weights = []
        for layer in self.layers:
            layer_params = {}
            for attr in ("Weights", "Biases", "W", "b"):
                if hasattr(layer, attr):
                    layer_params[attr] = copy.deepcopy(getattr(layer, attr))
            weights.append(layer_params)
        return weights

    def set_weights(self, weights):
        """Restore parameter model dari list yang dihasilkan get_weights_copy()."""
        for layer, layer_params in zip(self.layers, weights):
            for attr, value in layer_params.items():
                setattr(layer, attr, copy.deepcopy(value))