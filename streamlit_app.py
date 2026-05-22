from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import streamlit as st
from PIL import Image

from models.waste_cnn import WasteCNN
from predict import load_model_weights


MODEL_PATH = Path("outputs/weights/waste_cnn_20260518_173300.npz")
IMG_SIZE = 128
GRAYSCALE = False
CLASS_NAMES = ("inorganic", "organic")

CLASS_UI: Dict[str, Dict[str, str]] = {
    "organic": {
        "title": "Organic Waste",
        "subtitle": "Biodegradable material",
        "description": "Material ini cenderung mudah terurai secara alami oleh mikroorganisme.",
        "icon": "🍂",
        "accent": "#16a34a",
        "bg": "#f0fdf4",
        "border": "#bbf7d0",
    },
    "inorganic": {
        "title": "Inorganic Waste",
        "subtitle": "Non-biodegradable material",
        "description": "Material ini membutuhkan proses daur ulang atau penanganan khusus.",
        "icon": "♻️",
        "accent": "#0284c7",
        "bg": "#f0f9ff",
        "border": "#bae6fd",
    },
}


def inject_styles() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

        :root {
            --bg-primary: #fafbfc;
            --bg-card: #ffffff;
            --text-primary: #0f172a;
            --text-secondary: #475569;
            --text-muted: #94a3b8;
            --border: #e2e8f0;
            --border-light: #f1f5f9;
            --accent-green: #16a34a;
            --accent-blue: #0284c7;
            --radius-sm: 8px;
            --radius-md: 12px;
            --radius-lg: 16px;
            --radius-xl: 20px;
            --shadow-sm: 0 1px 2px rgba(0,0,0,0.04);
            --shadow-md: 0 4px 12px rgba(0,0,0,0.06);
            --shadow-lg: 0 8px 30px rgba(0,0,0,0.08);
        }

        * { font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important; }

        .stApp {
            background: var(--bg-primary);
            color: var(--text-primary);
        }

        [data-testid="stAppViewContainer"] > .main {
            padding-top: 0;
        }

        .block-container {
            max-width: 1080px;
            padding: 1.5rem 2rem 3rem;
        }

        /* ── Navbar ── */
        .navbar {
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 0.75rem 0;
            margin-bottom: 1.5rem;
            border-bottom: 1px solid var(--border);
        }
        .navbar-brand {
            display: flex;
            align-items: center;
            gap: 0.5rem;
            font-size: 1.15rem;
            font-weight: 700;
            color: var(--text-primary);
            letter-spacing: -0.02em;
        }
        .navbar-brand .logo-icon {
            width: 32px;
            height: 32px;
            border-radius: 8px;
            background: linear-gradient(135deg, #16a34a, #0284c7);
            display: flex;
            align-items: center;
            justify-content: center;
            color: #fff;
            font-size: 0.9rem;
        }
        .navbar-meta {
            display: flex;
            align-items: center;
            gap: 0.75rem;
        }
        .badge {
            display: inline-flex;
            align-items: center;
            gap: 0.3rem;
            padding: 0.25rem 0.65rem;
            border-radius: 999px;
            font-size: 0.72rem;
            font-weight: 600;
            letter-spacing: 0.01em;
        }
        .badge-green {
            background: #f0fdf4;
            color: #15803d;
            border: 1px solid #bbf7d0;
        }
        .badge-blue {
            background: #f0f9ff;
            color: #0369a1;
            border: 1px solid #bae6fd;
        }

        /* ── Hero ── */
        .hero-section {
            text-align: center;
            padding: 2rem 1rem 1.8rem;
            margin-bottom: 1.5rem;
        }
        .hero-section h1 {
            font-size: clamp(1.6rem, 3.5vw, 2.4rem);
            font-weight: 800;
            color: var(--text-primary);
            margin: 0 0 0.5rem;
            letter-spacing: -0.03em;
            line-height: 1.15;
        }
        .hero-section p {
            font-size: 1rem;
            color: var(--text-secondary);
            margin: 0 auto;
            max-width: 520px;
            line-height: 1.6;
        }

        /* ── Cards ── */
        .card {
            background: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: var(--radius-xl);
            padding: 1.5rem;
            box-shadow: var(--shadow-sm);
            transition: box-shadow 0.2s ease;
            height: 100%;
        }
        .card:hover {
            box-shadow: var(--shadow-md);
        }
        .card-header {
            display: flex;
            align-items: center;
            gap: 0.5rem;
            margin-bottom: 1rem;
            padding-bottom: 0.75rem;
            border-bottom: 1px solid var(--border-light);
        }
        .card-header-icon {
            width: 36px;
            height: 36px;
            border-radius: 10px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 1rem;
        }
        .card-header-text {
            font-size: 0.95rem;
            font-weight: 700;
            color: var(--text-primary);
            letter-spacing: -0.01em;
        }
        .card-header-sub {
            font-size: 0.75rem;
            color: var(--text-muted);
            font-weight: 500;
        }

        /* ── File uploader ── */
        [data-testid="stFileUploaderDropzone"] {
            border: 2px dashed #cbd5e1 !important;
            border-radius: var(--radius-md) !important;
            background: #f8fafc !important;
            padding: 1.5rem !important;
            transition: border-color 0.2s, background 0.2s;
        }
        [data-testid="stFileUploaderDropzone"]:hover {
            border-color: #94a3b8 !important;
            background: #f1f5f9 !important;
        }

        /* ── Placeholder ── */
        .empty-state {
            border: 1px dashed #d1d5db;
            border-radius: var(--radius-md);
            padding: 2.5rem 1.5rem;
            text-align: center;
            background: #fafbfc;
        }
        .empty-state .empty-icon {
            font-size: 2rem;
            margin-bottom: 0.5rem;
            opacity: 0.5;
        }
        .empty-state p {
            color: var(--text-muted);
            font-size: 0.88rem;
            margin: 0;
            line-height: 1.5;
        }

        /* ── Result Card ── */
        .result-card {
            border-radius: var(--radius-lg);
            padding: 1.25rem;
            margin-bottom: 1rem;
            display: flex;
            align-items: flex-start;
            gap: 0.85rem;
        }
        .result-icon {
            width: 44px;
            height: 44px;
            border-radius: 12px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 1.4rem;
            flex-shrink: 0;
        }
        .result-card h3 {
            margin: 0;
            font-size: 1.1rem;
            font-weight: 700;
            letter-spacing: -0.01em;
        }
        .result-card .result-sub {
            font-size: 0.82rem;
            margin: 0.15rem 0 0;
            opacity: 0.8;
        }
        .result-card .result-desc {
            font-size: 0.82rem;
            margin: 0.4rem 0 0;
            opacity: 0.7;
            line-height: 1.45;
        }

        /* ── Confidence ── */
        .confidence-section {
            margin: 0.75rem 0 0.5rem;
        }
        .confidence-label {
            display: flex;
            justify-content: space-between;
            align-items: baseline;
            margin-bottom: 0.35rem;
        }
        .confidence-label span:first-child {
            font-size: 0.82rem;
            font-weight: 600;
            color: var(--text-secondary);
        }
        .confidence-label span:last-child {
            font-size: 1.4rem;
            font-weight: 800;
            letter-spacing: -0.02em;
        }

        /* ── Distribution ── */
        .dist-title {
            font-size: 0.78rem;
            font-weight: 600;
            color: var(--text-muted);
            text-transform: uppercase;
            letter-spacing: 0.06em;
            margin: 1.25rem 0 0.6rem;
        }
        .dist-row {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 0.5rem 0;
            border-bottom: 1px solid var(--border-light);
        }
        .dist-row:last-child {
            border-bottom: none;
        }
        .dist-name {
            display: flex;
            align-items: center;
            gap: 0.4rem;
            font-size: 0.85rem;
            font-weight: 500;
            color: var(--text-primary);
        }
        .dist-dot {
            width: 8px;
            height: 8px;
            border-radius: 50%;
            flex-shrink: 0;
        }
        .dist-value {
            font-size: 0.85rem;
            font-weight: 700;
            color: var(--text-primary);
        }
        .dist-bar-bg {
            height: 6px;
            background: #f1f5f9;
            border-radius: 3px;
            margin-top: 0.3rem;
            overflow: hidden;
        }
        .dist-bar-fill {
            height: 100%;
            border-radius: 3px;
            transition: width 0.6s ease;
        }

        /* ── Footer ── */
        .footer {
            text-align: center;
            padding: 2rem 0 0.5rem;
            font-size: 0.75rem;
            color: var(--text-muted);
        }
        .footer a {
            color: var(--accent-blue);
            text-decoration: none;
        }

        /* ── Streamlit overrides ── */
        .stProgress > div > div > div > div {
            border-radius: 4px;
        }
        [data-testid="stImage"] {
            border-radius: var(--radius-md);
            overflow: hidden;
        }
        [data-testid="stImage"] img {
            border-radius: var(--radius-md);
        }

        /* Hide default streamlit elements */
        #MainMenu, footer, header {visibility: hidden;}

        @media (max-width: 768px) {
            .block-container {
                padding: 1rem 1rem 2rem;
            }
            .hero-section {
                padding: 1.5rem 0.5rem 1rem;
            }
            .card {
                padding: 1rem;
            }
            .navbar {
                flex-direction: column;
                gap: 0.5rem;
                align-items: flex-start;
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


def render_navbar() -> None:
    st.markdown(
        f"""
        <nav class="navbar">
            <div class="navbar-brand">
                <div class="logo-icon">🗑️</div>
                WasteVision
            </div>
            <div class="navbar-meta">
                <span class="badge badge-green">● Model Loaded</span>
                <span class="badge badge-blue">{MODEL_PATH.stem}</span>
            </div>
        </nav>
        """,
        unsafe_allow_html=True,
    )


def render_hero() -> None:
    st.markdown(
        """
        <section class="hero-section">
            <h1>Waste Image Classifier</h1>
            <p>Upload gambar sampah dan sistem akan mengklasifikasikan apakah termasuk <strong>organic</strong> atau <strong>inorganic</strong> secara instan.</p>
        </section>
        """,
        unsafe_allow_html=True,
    )


def render_footer() -> None:
    st.markdown(
        """
        <div class="footer">
            Built with Streamlit & Custom CNN · Waste Classification Project
        </div>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    st.set_page_config(
        page_title="WasteVision — Waste Classifier",
        page_icon="🗑️",
        layout="wide",
        initial_sidebar_state="collapsed",
    )

    inject_styles()
    render_navbar()
    render_hero()

    model_ready = True
    try:
        model = load_inference_model(str(MODEL_PATH))
    except Exception as exc:
        model_ready = False
        st.error(f"⚠️ Model gagal dimuat: {exc}")

    col_left, col_right = st.columns([1.1, 0.9], gap="large")

    # ── Left Panel: Upload ──
    with col_left:
        st.markdown(
            """
            <div class="card">
                <div class="card-header">
                    <div class="card-header-icon" style="background:#f0fdf4;">📁</div>
                    <div>
                        <div class="card-header-text">Upload Image</div>
                        <div class="card-header-sub">JPG, PNG, BMP, or WebP</div>
                    </div>
                </div>
            """,
            unsafe_allow_html=True,
        )

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
                caption=f"📎 {uploaded_file.name}  ·  {preview_image.size[0]}×{preview_image.size[1]} px",
                use_container_width=True,
            )
        else:
            st.markdown(
                """
                <div class="empty-state">
                    <div class="empty-icon">📷</div>
                    <p>Belum ada gambar dipilih.<br>Drag & drop atau klik untuk upload.</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("</div>", unsafe_allow_html=True)

    # ── Right Panel: Results ──
    with col_right:
        st.markdown(
            """
            <div class="card">
                <div class="card-header">
                    <div class="card-header-icon" style="background:#f0f9ff;">📊</div>
                    <div>
                        <div class="card-header-text">Prediction Result</div>
                        <div class="card-header-sub">Classification output</div>
                    </div>
                </div>
            """,
            unsafe_allow_html=True,
        )

        if uploaded_file is None:
            st.markdown(
                """
                <div class="empty-state">
                    <div class="empty-icon">🔍</div>
                    <p>Hasil prediksi akan muncul di sini<br>setelah gambar diunggah.</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
        elif not model_ready:
            st.warning("Model belum siap. Periksa path model lalu refresh halaman.")
        else:
            with st.spinner("Menganalisis gambar..."):
                pred_label, confidence, probs = run_inference(model, preview_image)

            ui = CLASS_UI[pred_label]

            # Result card
            st.markdown(
                f"""
                <div class="result-card" style="background:{ui['bg']}; border: 1px solid {ui['border']};">
                    <div class="result-icon" style="background:{ui['border']};">{ui['icon']}</div>
                    <div>
                        <h3 style="color:{ui['accent']};">{ui['title']}</h3>
                        <p class="result-sub" style="color:{ui['accent']};">{ui['subtitle']}</p>
                        <p class="result-desc" style="color:{ui['accent']};">{ui['description']}</p>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Confidence
            st.markdown(
                f"""
                <div class="confidence-section">
                    <div class="confidence-label">
                        <span>Confidence Score</span>
                        <span style="color:{ui['accent']};">{confidence * 100:.1f}%</span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.progress(float(confidence))

            # Distribution
            st.markdown('<div class="dist-title">Probability Distribution</div>', unsafe_allow_html=True)

            for class_name, probability in zip(CLASS_NAMES, probs):
                info = CLASS_UI[class_name]
                pct = probability * 100
                st.markdown(
                    f"""
                    <div class="dist-row">
                        <div class="dist-name">
                            <span class="dist-dot" style="background:{info['accent']};"></span>
                            {info['title']}
                        </div>
                        <span class="dist-value" style="color:{info['accent']};">{pct:.2f}%</span>
                    </div>
                    <div class="dist-bar-bg">
                        <div class="dist-bar-fill" style="width:{pct}%; background:{info['accent']};"></div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        st.markdown("</div>", unsafe_allow_html=True)

    render_footer()


if __name__ == "__main__":
    main()
