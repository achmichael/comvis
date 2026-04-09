from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import streamlit as st
from PIL import Image

from models.waste_cnn import WasteCNN
from predict import load_model_weights


MODEL_PATH = Path("outputs/weights/waste_cnn_20260409_150316.npz")
IMG_SIZE = 32
GRAYSCALE = False
CLASS_NAMES = ("inorganic", "organic")

CLASS_UI: Dict[str, Dict[str, str]] = {
    "organic": {
        "title": "Organic Waste",
        "subtitle": "Biodegradable material",
        "description": "Material cenderung mudah terurai secara alami.",
        "accent": "#2AA66A",
    },
    "inorganic": {
        "title": "Inorganic Waste",
        "subtitle": "Non-biodegradable material",
        "description": "Material cenderung membutuhkan proses daur ulang khusus.",
        "accent": "#1F6E8C",
    },
}


def inject_styles() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Sora:wght@400;600;700&family=Space+Grotesk:wght@500;700&display=swap');

        :root {
            --bg-soft: #f4fbf8;
            --bg-panel: #ffffff;
            --text-main: #173042;
            --text-sub: #4e6475;
            --line-soft: #d9e7e2;
            --accent: #2aa66a;
            --accent-2: #1f6e8c;
        }

        .stApp {
            background:
                radial-gradient(circle at 12% 5%, #dff5eb 0%, transparent 38%),
                radial-gradient(circle at 90% 0%, #d8edf6 0%, transparent 34%),
                linear-gradient(180deg, #f6fcfa 0%, #eef7f3 100%);
            color: var(--text-main);
        }

        [data-testid="stAppViewContainer"] > .main {
            padding-top: 1.2rem;
        }

        .block-container {
            max-width: 1100px;
            padding-top: 0.5rem;
            padding-bottom: 2rem;
        }

        .hero {
            background: linear-gradient(145deg, #ffffff 0%, #f5fffb 100%);
            border: 1px solid var(--line-soft);
            border-radius: 20px;
            padding: 1.3rem 1.5rem;
            box-shadow: 0 12px 30px rgba(30, 65, 56, 0.08);
            margin-bottom: 1rem;
        }

        .hero-tag {
            display: inline-block;
            font-family: "Space Grotesk", sans-serif;
            font-size: 0.75rem;
            font-weight: 700;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            color: #1f6e8c;
            background: #e3f2f8;
            border-radius: 999px;
            padding: 0.3rem 0.7rem;
            margin-bottom: 0.8rem;
        }

        .hero h1 {
            font-family: "Sora", sans-serif;
            font-size: clamp(1.5rem, 3vw, 2.2rem);
            margin: 0;
            color: #173042;
            line-height: 1.2;
        }

        .hero p {
            margin: 0.55rem 0 0;
            color: var(--text-sub);
            font-family: "Sora", sans-serif;
            font-size: 0.98rem;
        }

        .model-pill {
            margin-top: 0.9rem;
            display: inline-flex;
            align-items: center;
            gap: 0.4rem;
            padding: 0.35rem 0.75rem;
            border-radius: 999px;
            background: #ebfff4;
            color: #1d8f58;
            border: 1px solid #bcebd4;
            font-family: "Space Grotesk", sans-serif;
            font-size: 0.82rem;
            font-weight: 600;
        }

        .panel {
            background: var(--bg-panel);
            border: 1px solid var(--line-soft);
            border-radius: 18px;
            padding: 1rem;
            box-shadow: 0 10px 24px rgba(22, 56, 61, 0.06);
            min-height: 100%;
        }

        .panel-title {
            font-family: "Space Grotesk", sans-serif;
            font-size: 1rem;
            font-weight: 700;
            color: #19344a;
            margin-bottom: 0.6rem;
        }

        [data-testid="stFileUploaderDropzone"] {
            border: 1.5px dashed #9ccfc2 !important;
            border-radius: 14px !important;
            background: #f8fffc !important;
            padding: 1.2rem !important;
        }

        .placeholder {
            border: 1px dashed #c7d8d2;
            border-radius: 12px;
            padding: 1rem;
            text-align: center;
            font-family: "Sora", sans-serif;
            color: #67808f;
            background: #fcfffe;
        }

        .prediction-card {
            border-radius: 14px;
            border: 1px solid #e2ece8;
            background: #fbfffd;
            padding: 1rem;
            margin-bottom: 0.6rem;
        }

        .prediction-card h3 {
            margin: 0;
            font-family: "Sora", sans-serif;
            font-size: 1.26rem;
            color: #143347;
        }

        .prediction-card p {
            margin: 0.25rem 0 0;
            font-family: "Sora", sans-serif;
            color: #587284;
            font-size: 0.93rem;
        }

        .score-row {
            display: flex;
            justify-content: space-between;
            margin: 0.35rem 0 0.2rem;
            font-family: "Sora", sans-serif;
            color: #173042;
            font-size: 0.9rem;
        }

        .caption-soft {
            color: #68818f;
            font-family: "Sora", sans-serif;
            font-size: 0.85rem;
        }

        @media (max-width: 900px) {
            .hero {
                padding: 1rem;
            }
            .panel {
                padding: 0.85rem;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp_scores = np.exp(shifted)
    return exp_scores / np.sum(exp_scores, axis=1, keepdims=True)


def preprocess_uploaded_image(image: Image.Image) -> np.ndarray:
    mode = "L" if GRAYSCALE else "RGB"
    model_image = image.convert(mode).resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)

    arr = np.asarray(model_image, dtype=np.float32) / 255.0
    if GRAYSCALE:
        arr = arr[np.newaxis, :, :]
    else:
        arr = np.transpose(arr, (2, 0, 1))

    return arr.astype(np.float32)


@st.cache_resource
def load_inference_model(model_path: str) -> WasteCNN:
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"Model tidak ditemukan di path: {path}")

    model = WasteCNN(
        input_size=IMG_SIZE,
        in_channels=1 if GRAYSCALE else 3,
        num_classes=len(CLASS_NAMES),
    )
    load_model_weights(model, path)
    return model


def run_inference(model: WasteCNN, image: Image.Image) -> Tuple[str, float, np.ndarray]:
    x = np.expand_dims(preprocess_uploaded_image(image), axis=0)
    logits = model.forward(x)
    probs = softmax(logits)[0]

    pred_idx = int(np.argmax(probs))
    pred_label = CLASS_NAMES[pred_idx]
    confidence = float(probs[pred_idx])
    return pred_label, confidence, probs


def render_hero() -> None:
    st.markdown(
        f"""
        <section class="hero">
            <span class="hero-tag">Computer Vision Demo</span>
            <h1>Waste Image Classifier</h1>
            <p>Upload satu gambar sampah dan sistem akan memprediksi apakah termasuk organic atau inorganic secara instan.</p>
            <span class="model-pill">Model aktif: {MODEL_PATH.name}</span>
        </section>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    st.set_page_config(
        page_title="Waste Classification App",
        page_icon="WC",
        layout="wide",
        initial_sidebar_state="collapsed",
    )

    inject_styles()
    render_hero()

    model_ready = True
    try:
        model = load_inference_model(str(MODEL_PATH))
    except Exception as exc:
        model_ready = False
        st.error(f"Model gagal dimuat: {exc}")

    col_left, col_right = st.columns([1.08, 0.92], gap="large")

    with col_left:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown('<div class="panel-title">Upload Image</div>', unsafe_allow_html=True)

        uploaded_file = st.file_uploader(
            "Upload image",
            type=["jpg", "jpeg", "png", "bmp", "webp"],
            label_visibility="collapsed",
        )

        preview_image = None
        if uploaded_file is not None:
            preview_image = Image.open(uploaded_file).convert("RGB")
            st.image(
                preview_image,
                caption=f"{uploaded_file.name} | {preview_image.size[0]} x {preview_image.size[1]}",
                use_container_width=True,
            )
        else:
            st.markdown(
                '<div class="placeholder">Belum ada gambar dipilih. Silakan upload file untuk mulai prediksi.</div>',
                unsafe_allow_html=True,
            )

        st.markdown("</div>", unsafe_allow_html=True)

    with col_right:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown('<div class="panel-title">Prediction Result</div>', unsafe_allow_html=True)

        if uploaded_file is None:
            st.markdown(
                '<div class="placeholder">Hasil prediksi akan muncul di sini setelah gambar diunggah.</div>',
                unsafe_allow_html=True,
            )
        elif not model_ready:
            st.warning("Model belum siap, periksa path model lalu refresh halaman.")
        else:
            with st.spinner("Menganalisis gambar..."):
                pred_label, confidence, probs = run_inference(model, preview_image)

            ui = CLASS_UI[pred_label]
            st.markdown(
                f"""
                <div class="prediction-card" style="border-left: 6px solid {ui['accent']};">
                    <h3>{ui['title']}</h3>
                    <p>{ui['subtitle']}</p>
                    <p>{ui['description']}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.markdown(f"**Confidence:** {confidence * 100:.2f}%")
            st.progress(float(confidence))
            st.markdown('<p class="caption-soft">Distribusi probabilitas per kelas</p>', unsafe_allow_html=True)

            for class_name, probability in zip(CLASS_NAMES, probs):
                info = CLASS_UI[class_name]
                st.markdown(
                    f"<div class=\"score-row\"><span>{info['title']}</span><span>{probability * 100:.2f}%</span></div>",
                    unsafe_allow_html=True,
                )
                st.progress(float(probability))

        st.markdown("</div>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
