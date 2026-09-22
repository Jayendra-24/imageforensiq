"""
Forensic Image Forgery Detector — app.py
Run with:  streamlit run app.py

FIXES vs original:
  - Uses the saved scaler from model_bundle (was missing → wrong feature scale)
  - Uses calibrated probability threshold (configurable)
  - Shows per-technique scores with clear authentic/forged indicators
  - Uses the same 12-feature extraction as prepare_data.py / train_model.py
"""

import streamlit as st
import numpy as np
import cv2
import joblib
import io
from PIL import Image

# ─── PAGE CONFIG ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Forensic Image Detector",
    page_icon="🔍",
    layout="wide"
)

# ─── FORENSIC FUNCTIONS (must match prepare_data.py exactly) ─────────────────

def ela_analysis(image_array, quality=92):
    img_pil = Image.fromarray(cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB))
    buffer  = io.BytesIO()
    img_pil.save(buffer, format='JPEG', quality=quality)
    buffer.seek(0)
    recompressed    = np.array(Image.open(buffer))
    original_rgb    = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
    ela             = cv2.absdiff(original_rgb, recompressed).astype(np.float32)
    ela_gray        = np.mean(ela, axis=2)

    mean_ela = float(np.mean(ela_gray))
    max_ela  = float(np.max(ela_gray))

    h, w = ela_gray.shape
    block_size = 64
    block_means = []
    for y in range(0, h - block_size, block_size):
        for x in range(0, w - block_size, block_size):
            block = ela_gray[y:y + block_size, x:x + block_size]
            block_means.append(np.mean(block))

    block_means = np.array(block_means)
    ela_cv = float(np.std(block_means) / (np.mean(block_means) + 1e-6))
    ela_max_median_ratio = float(np.max(block_means) / (np.median(block_means) + 1e-6))

    # Build ELA visualization
    ela_vis = np.clip(ela_gray * 10, 0, 255).astype(np.uint8)
    ela_vis_color = cv2.applyColorMap(ela_vis, cv2.COLORMAP_JET)

    return {
        'mean': mean_ela, 'max': max_ela,
        'block_cv': ela_cv, 'max_med_ratio': ela_max_median_ratio,
        'visualization': ela_vis_color
    }


def noise_analysis(image_array):
    gray      = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY).astype(np.float32)
    laplacian = cv2.Laplacian(gray, cv2.CV_32F)
    noise_map = np.abs(laplacian)

    global_mean = float(np.mean(noise_map))
    h, w = noise_map.shape
    block_size = 64
    block_stds = []
    for y in range(0, h - block_size, block_size):
        for x in range(0, w - block_size, block_size):
            block = noise_map[y:y + block_size, x:x + block_size]
            block_stds.append(np.std(block))

    block_stds  = np.array(block_stds)
    noise_cv    = float(np.std(block_stds) / (np.mean(block_stds) + 1e-6))
    noise_range = float(np.max(block_stds) - np.min(block_stds))

    # Noise visualization
    noise_vis = np.clip(noise_map, 0, 255).astype(np.uint8)

    return {
        'global_mean': global_mean,
        'block_cv': noise_cv,
        'block_range': noise_range,
        'visualization': noise_vis
    }


def dct_analysis(image_array):
    gray = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY).astype(np.float32)
    h, w = gray.shape
    block_size = 8
    high_freq_ratios = []

    for y in range(0, h - block_size, block_size):
        for x in range(0, w - block_size, block_size):
            block = gray[y:y + block_size, x:x + block_size]
            dct_block  = cv2.dct(block)
            high_freq  = np.sum(np.abs(dct_block[4:, 4:]))
            total      = np.sum(np.abs(dct_block)) + 1e-6
            high_freq_ratios.append(high_freq / total)

    high_freq_ratios = np.array(high_freq_ratios)
    return {
        'variance': float(np.var(high_freq_ratios)),
        'mean_hf':  float(np.mean(high_freq_ratios))
    }


def statistical_analysis(image_array):
    gray            = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)
    pixels          = gray.flatten()
    pixels_nonzero  = pixels[pixels > 0]

    first_digits = [
        int(str(int(p))[0])
        for p in pixels_nonzero[::10]
        if 1 <= int(str(int(p))[0]) <= 9
    ]

    if len(first_digits) == 0:
        return {'benford_deviation': 0.0, 'entropy': 0.0}

    digit_counts = np.zeros(9)
    for d in first_digits:
        digit_counts[d - 1] += 1
    digit_freq = digit_counts / len(first_digits)
    benford    = np.array([np.log10(1 + 1 / d) for d in range(1, 10)])
    b_dev      = float(np.mean(np.abs(digit_freq - benford)))

    hist    = np.bincount(gray.flatten(), minlength=256).astype(np.float64)
    hist    = hist[hist > 0]
    prob    = hist / hist.sum()
    entropy = float(-np.sum(prob * np.log2(prob)))

    return {'benford_deviation': b_dev, 'entropy': entropy}


def texture_analysis(image_array):
    gray = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    block_size = 64
    lbp_stds = []

    for y in range(0, h - block_size, block_size):
        for x in range(0, w - block_size, block_size):
            block    = gray[y:y + block_size, x:x + block_size].astype(np.float32)
            gx       = cv2.Sobel(block, cv2.CV_32F, 1, 0, ksize=3)
            gy       = cv2.Sobel(block, cv2.CV_32F, 0, 1, ksize=3)
            grad_mag = np.sqrt(gx**2 + gy**2)
            lbp_stds.append(np.std(grad_mag))

    lbp_stds   = np.array(lbp_stds)
    texture_cv = float(np.std(lbp_stds) / (np.mean(lbp_stds) + 1e-6))
    return {'texture_cv': texture_cv}


def extract_features(image_array):
    """Extract all 12 features from an image array (BGR, 512×512)."""
    ela     = ela_analysis(image_array)
    noise   = noise_analysis(image_array)
    dct     = dct_analysis(image_array)
    stats   = statistical_analysis(image_array)
    texture = texture_analysis(image_array)

    features = [
        ela['mean'],              ela['max'],
        ela['block_cv'],          ela['max_med_ratio'],
        noise['global_mean'],     noise['block_cv'],     noise['block_range'],
        dct['variance'],          dct['mean_hf'],
        stats['benford_deviation'], stats['entropy'],
        texture['texture_cv'],
    ]
    return features, ela, noise


# ─── LOAD MODEL ──────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    try:
        bundle = joblib.load('forensic_model.pkl')
        # Support both old (bare model) and new (bundle dict) format
        if isinstance(bundle, dict):
            return bundle['model'], bundle.get('scaler'), bundle.get('threshold', 0.5)
        else:
            # Old format — no scaler saved, warn the user
            st.warning("⚠ Old model format detected. Retrain with the new train_model.py for best accuracy.")
            return bundle, None, 0.5
    except FileNotFoundError:
        return None, None, 0.5


model, scaler, threshold = load_model()

# ─── UI ──────────────────────────────────────────────────────────────────────
st.title("🔍 Forensic Image Forgery Detector")
st.markdown("Upload an image to analyze it for signs of digital manipulation.")

if model is None:
    st.error("❌ `forensic_model.pkl` not found. Run `train_model.py` first.")
    st.stop()

col1, col2 = st.columns([1, 1])

with col1:
    uploaded = st.file_uploader(
        "Upload image", type=['jpg', 'jpeg', 'png', 'bmp', 'tif'],
        label_visibility='collapsed'
    )

if uploaded:
    file_bytes = np.frombuffer(uploaded.read(), np.uint8)
    image_bgr  = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if image_bgr is None:
        st.error("Could not read the uploaded image.")
        st.stop()

    image_bgr_resized = cv2.resize(image_bgr, (512, 512))

    # Show original
    with col1:
        st.image(
            cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB),
            caption="Uploaded Image", use_container_width=True
        )

    # ── Run analysis ──────────────────────────────────────────────────────
    with st.spinner("Running forensic analysis…"):
        features, ela_result, noise_result = extract_features(image_bgr_resized)

    feat_array = np.array(features).reshape(1, -1)

    # Apply scaler if available (CRITICAL FIX)
    if scaler is not None:
        feat_array = scaler.transform(feat_array)

    prob_forged = model.predict_proba(feat_array)[0][1]
    is_forged   = prob_forged >= threshold

    # ── Verdict ───────────────────────────────────────────────────────────
    with col2:
        confidence = prob_forged if is_forged else (1 - prob_forged)
        confidence_pct = int(confidence * 100)

        if is_forged:
            st.error(f"## ⚠ Likely FORGED  ({confidence_pct}% confidence)")
        else:
            st.success(f"## ✅ Likely AUTHENTIC  ({confidence_pct}% confidence)")

        st.progress(int(prob_forged * 100), text=f"Forgery probability: {prob_forged*100:.1f}%")

    # ── Detailed results ──────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("📊 Forensic Analysis Details")

    r1, r2, r3, r4 = st.columns(4)

    with r1:
        st.markdown("**ELA Analysis**")
        st.metric("Mean ELA",      f"{ela_result['mean']:.2f}")
        st.metric("Block Variance", f"{ela_result['block_cv']:.3f}")
        ela_vis_rgb = cv2.cvtColor(ela_result['visualization'], cv2.COLOR_BGR2RGB)
        st.image(ela_vis_rgb, caption="ELA Map (brighter = suspicious)", use_container_width=True)

    with r2:
        st.markdown("**Noise Analysis**")
        st.metric("Global Noise",   f"{noise_result['global_mean']:.2f}")
        st.metric("Noise Variance", f"{noise_result['block_cv']:.3f}")
        noise_vis_rgb = cv2.cvtColor(
            cv2.applyColorMap(
                np.clip(noise_result['visualization'], 0, 255).astype(np.uint8),
                cv2.COLORMAP_BONE
            ), cv2.COLOR_BGR2RGB
        )
        st.image(noise_vis_rgb, caption="Noise Map", use_container_width=True)

    with r3:
        st.markdown("**DCT Analysis**")
        dct_res = dct_analysis(image_bgr_resized)
        st.metric("DCT Variance", f"{dct_res['variance']:.5f}")
        st.metric("High Freq",    f"{dct_res['mean_hf']:.3f}")

    with r4:
        st.markdown("**Statistical**")
        stat_res = statistical_analysis(image_bgr_resized)
        st.metric("Benford Dev", f"{stat_res['benford_deviation']:.5f}")
        st.metric("Entropy",     f"{stat_res['entropy']:.2f}")

    # ── Interpretation ────────────────────────────────────────────────────
    st.markdown("---")
    with st.expander("📖 How to interpret these results"):
        st.markdown("""
        | Indicator | Authentic | Forged |
        |---|---|---|
        | **ELA Block Variance** | Low (uniform compression) | High (inconsistent regions) |
        | **Noise Variance** | Low (same camera sensor) | High (different sources blended) |
        | **DCT Variance** | Low (single compression) | High (double-compressed regions) |
        | **Benford Deviation** | Near 0 (natural distribution) | Elevated (unnatural pixel stats) |

        **Note:** No single technique is 100% reliable. The AI model combines all indicators
        for a final decision. Always corroborate results with professional tools for legal purposes.
        """)
