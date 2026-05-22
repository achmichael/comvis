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
        "description": "Material ini mudah terurai secara alami oleh mikroorganisme.",
        "icon": "🍂",
        "accent": "#16a34a",
        "accent_dark": "#15803d",
        "bg": "#f0fdf4",
        "border": "#bbf7d0",
    },
    "inorganic": {
        "title": "Inorganic Waste",
        "subtitle": "Non-biodegradable material",
        "description": "Material ini membutuhkan proses daur ulang atau penanganan khusus.",
        "icon": "♻️",
        "accent": "#0284c7",
        "accent_dark": "#0369a1",
        "bg": "#f0f9ff",
        "border": "#bae6fd",
    },
}


def inject_styles() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

        /* ══════════════════════════════════════
           GLOBAL RESET & BASE
           ══════════════════════════════════════ */
        :root {
            --bg: #f4f6f8;
            --card: #ffffff;
            --text-1: #1e293b;
            --text-2: #475569;
            --text-3: #94a3b8;
            --border: #e2e8f0;
            --border-light: #f1f5f9;
            --green: #16a34a;
            --blue: #0284c7;
            --r: 14px;
        }

        html, body, .stApp,
        .stApp * {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif !important;
        }

        .stApp {
            background: var(--bg) !important;
        }

        /* reduce top padding */
        [data-testid="stAppViewContainer"] > .main { padding-top: 0; }
        .block-container { max-width: 1060px; padding: 1rem 1.5rem 2rem; }

        /* hide streamlit chrome */
        #MainMenu, footer, header { visibility: hidden; }
        [data-testid="stDecoration"] { display: none; }

        /* ══════════════════════════════════════
           STREAMLIT WIDGET OVERRIDES
           Fix text visibility on buttons, uploader, etc.
           ══════════════════════════════════════ */

        /* --- Buttons --- */
        .stButton > button,
        button[kind="secondary"],
        [data-testid="stBaseButton-secondary"] {
            background: #ffffff !important;
            color: #1e293b !important;
            border: 1px solid #d1d5db !important;
            border-radius: 8px !important;
            font-weight: 500 !important;
            font-size: 0.85rem !important;
            padding: 0.4rem 1rem !important;
            transition: all 0.15s ease !important;
            box-shadow: 0 1px 2px rgba(0,0,0,0.04) !important;
        }
        .stButton > button:hover,
        button[kind="secondary"]:hover,
        [data-testid="stBaseButton-secondary"]:hover {
            background: #f8fafc !important;
            border-color: #9ca3af !important;
            color: #0f172a !important;
        }
        .stButton > button:active,
        [data-testid="stBaseButton-secondary"]:active {
            background: #f1f5f9 !important;
        }

        /* --- File uploader container --- */
        [data-testid="stFileUploader"] {
            margin-bottom: 0.5rem;
        }
        /* Hide Streamlit's built-in uploader label (prevents double label) */
        [data-testid="stFileUploader"] > label,
        [data-testid="stFileUploader"] > div[data-testid="stWidgetLabel"],
        [data-testid="stFileUploader"] [data-testid="stWidgetLabel"] {
            display: none !important;
        }

        /* Dropzone — high contrast, clearly visible */
        [data-testid="stFileUploaderDropzone"] {
            border: 2.5px dashed #94a3b8 !important;
            border-radius: 14px !important;
            background: #ffffff !important;
            padding: 1.8rem 1.5rem !important;
            transition: all 0.2s ease !important;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            gap: 0.4rem;
        }
        [data-testid="stFileUploaderDropzone"]:hover {
            border-color: #16a34a !important;
            background: #f7fdf9 !important;
        }

        /* Instructional text inside dropzone — keep visible, clean style */
        [data-testid="stFileUploaderDropzone"] span,
        [data-testid="stFileUploaderDropzone"] div {
            color: #64748b !important;
            font-size: 0.85rem !important;
            font-weight: 400 !important;
            text-align: center !important;
        }
        [data-testid="stFileUploaderDropzone"] small {
            color: #94a3b8 !important;
            font-size: 0.75rem !important;
        }

        /* Browse files button — prominent dark button */
        [data-testid="stFileUploaderDropzone"] button,
        [data-testid="stFileUploaderDropzone"] [data-testid="stBaseButton-secondary"],
        [data-testid="stFileUploaderDropzone"] [data-testid="baseButton-secondary"] {
            background: #0f172a !important;
            color: #ffffff !important;
            border: none !important;
            border-radius: 8px !important;
            font-weight: 600 !important;
            font-size: 0.85rem !important;
            padding: 0.5rem 1.4rem !important;
            cursor: pointer !important;
            transition: background 0.15s ease !important;
            margin-top: 0.3rem !important;
        }
        [data-testid="stFileUploaderDropzone"] button:hover,
        [data-testid="stFileUploaderDropzone"] [data-testid="stBaseButton-secondary"]:hover,
        [data-testid="stFileUploaderDropzone"] [data-testid="baseButton-secondary"]:hover {
            background: #1e293b !important;
            color: #ffffff !important;
        }
        [data-testid="stFileUploaderDropzone"] button:focus {
            outline: 2px solid #16a34a !important;
            outline-offset: 2px !important;
        }

        /* Uploaded file name chip */
        [data-testid="stFileUploaderFile"] {
            color: #1e293b !important;
            background: #f8fafc !important;
            border: 1px solid #e2e8f0 !important;
            border-radius: 8px !important;
            padding: 0.4rem 0.6rem !important;
        }
        [data-testid="stFileUploaderFile"] small,
        [data-testid="stFileUploaderFile"] span {
            color: #475569 !important;
        }
        [data-testid="stFileUploaderDeleteBtn"] {
            color: #ef4444 !important;
        }
        [data-testid="stFileUploaderDeleteBtn"]:hover {
            color: #dc2626 !important;
            background: #fef2f2 !important;
            border-radius: 6px !important;
        }

        /* --- Spinner --- */
        .stSpinner > div { color: #475569 !important; }

        /* --- Progress bar --- */
        .stProgress > div > div > div {
            background: #e2e8f0 !important;
            border-radius: 6px !important;
        }
        .stProgress > div > div > div > div {
            border-radius: 6px !important;
            background: linear-gradient(90deg, var(--green), var(--blue)) !important;
        }

        /* --- Image captions --- */
        [data-testid="stImage"] {
            border-radius: 10px;
            overflow: hidden;
        }
        [data-testid="stImage"] img {
            border-radius: 10px;
        }
        [data-testid="stCaptionContainer"] {
            color: #64748b !important;
            font-size: 0.8rem !important;
        }

        /* --- Warning/Error boxes --- */
        [data-testid="stAlert"] {
            border-radius: 10px !important;
        }

        /* --- Markdown text --- */
        .stMarkdown p, .stMarkdown span {
            color: #1e293b;
        }

        /* ══════════════════════════════════════
           CUSTOM COMPONENTS
           ══════════════════════════════════════ */

        /* ── Topbar ── */
        .topbar {
            display: flex;
            align-items: center;
            justify-content: space-between;
            flex-wrap: wrap;
            gap: 0.5rem;
            padding: 0.85rem 0;
            margin-bottom: 0.5rem;
            border-bottom: 1px solid var(--border);
        }
        .topbar-brand {
            display: flex;
            align-items: center;
            gap: 0.55rem;
            font-size: 1.1rem;
            font-weight: 700;
            color: var(--text-1);
            letter-spacing: -0.02em;
        }
        .topbar-logo {
            width: 30px; height: 30px;
            border-radius: 8px;
            background: linear-gradient(135deg, #16a34a 0%, #0ea5e9 100%);
            display: grid; place-items: center;
            color: #fff; font-size: 0.85rem;
        }
        .topbar-right {
            display: flex; align-items: center; gap: 0.5rem;
        }
        .pill {
            display: inline-flex; align-items: center; gap: 0.3rem;
            padding: 0.2rem 0.6rem; border-radius: 999px;
            font-size: 0.7rem; font-weight: 600;
        }
        .pill-ok {
            background: #ecfdf5; color: #047857; border: 1px solid #a7f3d0;
        }
        .pill-info {
            background: #eff6ff; color: #1d4ed8; border: 1px solid #bfdbfe;
        }

        /* ── Hero ── */
        .hero {
            text-align: center;
            padding: 2.2rem 1rem 1.6rem;
        }
        .hero h1 {
            font-size: clamp(1.5rem, 3.2vw, 2.2rem);
            font-weight: 800; letter-spacing: -0.03em;
            color: var(--text-1); margin: 0 0 0.4rem; line-height: 1.15;
        }
        .hero h1 .gradient-text {
            background: linear-gradient(135deg, #16a34a, #0284c7);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }
        .hero p {
            color: var(--text-2); font-size: 0.95rem;
            max-width: 480px; margin: 0 auto; line-height: 1.55;
        }

        /* ── Section label ── */
        .section-label {
            display: flex; align-items: center; gap: 0.5rem;
            font-size: 0.9rem; font-weight: 700; color: var(--text-1);
            margin-bottom: 0.85rem;
            padding: 0.7rem 0.9rem;
            background: #ffffff;
            border: 1px solid var(--border);
            border-radius: 12px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.04);
        }
        .section-label .sl-icon {
            font-size: 1.05rem;
        }
        .section-sub {
            font-size: 0.72rem; font-weight: 500; color: var(--text-3);
            margin-left: auto;
        }

        /* ── Empty state ── */
        .empty-box {
            border: 1.5px dashed #cbd5e1;
            border-radius: 12px;
            padding: 2rem 1rem;
            text-align: center;
            background: #fafbfc;
        }
        .empty-box .e-icon { font-size: 1.8rem; margin-bottom: 0.4rem; opacity: 0.45; }
        .empty-box p { color: #94a3b8; font-size: 0.84rem; margin: 0; line-height: 1.5; }

        /* ── Result banner ── */
        .result-banner {
            display: flex; align-items: flex-start; gap: 0.9rem;
            border-radius: 14px;
            padding: 1.15rem 1.2rem;
            margin-bottom: 0.75rem;
        }
        .rb-icon {
            width: 46px; height: 46px; border-radius: 12px;
            display: grid; place-items: center;
            font-size: 1.5rem; flex-shrink: 0;
        }
        .rb-body h3 {
            margin: 0; font-size: 1.05rem; font-weight: 700;
            letter-spacing: -0.01em;
        }
        .rb-body .rb-sub {
            font-size: 0.78rem; margin: 0.1rem 0 0; font-weight: 500;
        }
        .rb-body .rb-desc {
            font-size: 0.78rem; margin: 0.35rem 0 0; line-height: 1.45;
            opacity: 0.75;
        }

        /* ── Confidence big number ── */
        .conf-box {
            display: flex; justify-content: space-between;
            align-items: baseline;
            margin: 0.6rem 0 0.25rem;
        }
        .conf-label {
            font-size: 0.78rem; font-weight: 600; color: var(--text-2);
            text-transform: uppercase; letter-spacing: 0.04em;
        }
        .conf-value {
            font-size: 1.6rem; font-weight: 800; letter-spacing: -0.03em;
        }

        /* ── Distribution ── */
        .dist-head {
            font-size: 0.72rem; font-weight: 600; color: var(--text-3);
            text-transform: uppercase; letter-spacing: 0.06em;
            margin: 1rem 0 0.5rem;
        }
        .dist-item {
            margin-bottom: 0.65rem;
        }
        .dist-item-row {
            display: flex; justify-content: space-between; align-items: center;
            margin-bottom: 0.25rem;
        }
        .di-name {
            display: flex; align-items: center; gap: 0.35rem;
            font-size: 0.82rem; font-weight: 500; color: var(--text-1);
        }
        .di-dot {
            width: 8px; height: 8px; border-radius: 50%; flex-shrink: 0;
        }
        .di-pct {
            font-size: 0.82rem; font-weight: 700;
        }
        .di-bar {
            height: 7px; background: #f1f5f9; border-radius: 4px;
            overflow: hidden;
        }
        .di-bar-fill {
            height: 100%; border-radius: 4px;
            transition: width 0.5s ease;
        }

        /* ── Separator ── */
        .sep { height: 1px; background: var(--border-light); margin: 0.9rem 0; }

        /* ── Footer ── */
        .app-footer {
            text-align: center; padding: 1.8rem 0 0.3rem;
            font-size: 0.72rem; color: var(--text-3);
        }

        /* ── Columns gap fix ── */
        [data-testid="stHorizontalBlock"] { gap: 1.5rem; }

        /* ── Responsive ── */
        @media (max-width: 768px) {
            .block-container { padding: 0.75rem 0.75rem 1.5rem; }
            .hero { padding: 1.5rem 0.5rem 1rem; }
            .topbar { flex-direction: column; align-items: flex-start; }
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


# ────────────────────────────────────────
# UI RENDERERS
# ────────────────────────────────────────

def render_topbar() -> None:
    st.markdown(
        f"""
        <div class="topbar">
            <div class="topbar-brand">
                <div class="topbar-logo">♻️</div>
                WasteVision
            </div>
            <div class="topbar-right">
                <span class="pill pill-ok">● Ready</span>
                <span class="pill pill-info">CNN Model</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_hero() -> None:
    st.markdown(
        """
        <div class="hero">
            <h1>Waste Image <span class="gradient-text">Classifier</span></h1>
            <p>Upload gambar sampah — sistem mengklasifikasikan <strong>organic</strong> atau <strong>inorganic</strong> secara instan.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_empty(icon: str, text: str) -> None:
    st.markdown(
        f"""
        <div class="empty-box">
            <div class="e-icon">{icon}</div>
            <p>{text}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_result(pred_label: str, confidence: float, probs: np.ndarray) -> None:
    ui = CLASS_UI[pred_label]

    # Result banner
    st.markdown(
        f"""
        <div class="result-banner" style="background:{ui['bg']}; border:1px solid {ui['border']};">
            <div class="rb-icon" style="background:{ui['border']};">{ui['icon']}</div>
            <div class="rb-body">
                <h3 style="color:{ui['accent_dark']};">{ui['title']}</h3>
                <p class="rb-sub" style="color:{ui['accent']};">{ui['subtitle']}</p>
                <p class="rb-desc" style="color:{ui['accent_dark']};">{ui['description']}</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Confidence score
    st.markdown(
        f"""
        <div class="conf-box">
            <span class="conf-label">Confidence</span>
            <span class="conf-value" style="color:{ui['accent_dark']};">{confidence * 100:.1f}%</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.progress(float(confidence))

    # Separator + distribution
    st.markdown('<div class="sep"></div>', unsafe_allow_html=True)
    st.markdown('<div class="dist-head">Probability Distribution</div>', unsafe_allow_html=True)

    for cls, prob in zip(CLASS_NAMES, probs):
        info = CLASS_UI[cls]
        pct = prob * 100
        st.markdown(
            f"""
            <div class="dist-item">
                <div class="dist-item-row">
                    <span class="di-name">
                        <span class="di-dot" style="background:{info['accent']};"></span>
                        {info['title']}
                    </span>
                    <span class="di-pct" style="color:{info['accent_dark']};">{pct:.2f}%</span>
                </div>
                <div class="di-bar">
                    <div class="di-bar-fill" style="width:{pct:.1f}%; background:{info['accent']};"></div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


# ────────────────────────────────────────
# MAIN
# ────────────────────────────────────────

def main() -> None:
    st.set_page_config(
        page_title="WasteVision — Classifier",
        page_icon="♻️",
        layout="wide",
        initial_sidebar_state="collapsed",
    )

    inject_styles()
    render_topbar()
    render_hero()

    # Load model
    model_ready = True
    try:
        model = load_inference_model(str(MODEL_PATH))
    except Exception as exc:
        model_ready = False
        st.error(f"⚠️ Model gagal dimuat: {exc}")

    # Two-column layout
    col_left, col_right = st.columns([1.1, 0.9], gap="large")

    # ── LEFT: Upload ──
    with col_left:
        st.markdown(
            """
            <div class="section-label">
                <span class="sl-icon">📁</span> Upload Image
                <span class="section-sub">JPG · PNG · BMP · WebP</span>
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
            render_empty("📷", "Belum ada gambar dipilih.<br>Drag & drop atau klik <b>Browse files</b>.")

    # ── RIGHT: Results ──
    with col_right:
        st.markdown(
            """
            <div class="section-label">
                <span class="sl-icon">📊</span> Prediction Result
                <span class="section-sub">Classification output</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if uploaded_file is None:
            render_empty("🔍", "Hasil prediksi muncul di sini<br>setelah gambar diunggah.")
        elif not model_ready:
            st.warning("Model belum siap. Periksa path model lalu refresh halaman.")
        else:
            with st.spinner("Menganalisis gambar..."):
                pred_label, confidence, probs = run_inference(model, preview_image)
            render_result(pred_label, confidence, probs)

    # Footer
    st.markdown(
        '<div class="app-footer">Built with Streamlit & Custom CNN · WasteVision</div>',
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
