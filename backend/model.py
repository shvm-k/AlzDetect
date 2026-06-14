"""
Model loading, preprocessing, inference and Grad-CAM for AlzDetect.

The architecture and preprocessing here mirror the research notebook exactly:
  - Input: 128x128x3, rescaled by 1/255
  - MobileNetV2 (imagenet) -> GAP -> Dense(128, relu) -> Dropout(0.3) -> Dense(4, softmax)
  - Class order matches the notebook's `dataset_classes` dict insertion order.

If a trained weights file is found in models/, the real model is used.
Otherwise the app runs in clearly-labelled DEMO MODE so the full pipeline
(upload -> preprocess -> predict -> heatmap) can still be demonstrated.
"""

import hashlib
import io
import os

import numpy as np
from PIL import Image

# Class order MUST match the notebook's dataset_classes dict.
CLASS_NAMES = [
    "Mild Dementia",
    "Moderate Dementia",
    "Very mild Dementia",
    "Non Demented",
]

IMG_SIZE = 128
MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")

# Lazily-populated globals.
_model = None
_demo_mode = True
_last_conv_layer = "Conv_1"  # final conv layer name in MobileNetV2


def _download_weights(url):
    """Download weights from MODEL_URL into MODELS_DIR. Returns path or None."""
    import urllib.request

    os.makedirs(MODELS_DIR, exist_ok=True)
    ext = ".h5" if url.split("?")[0].endswith(".h5") else ".keras"
    dest = os.path.join(MODELS_DIR, "downloaded_model" + ext)
    if os.path.exists(dest):
        return dest
    try:
        print(f"[AlzDetect] Downloading model from {url} ...")
        urllib.request.urlretrieve(url, dest)
        print(f"[AlzDetect] Saved model to {dest}")
        return dest
    except Exception as exc:  # pragma: no cover - defensive
        print(f"[AlzDetect] Model download failed: {exc}")
        return None


def _find_weights():
    """Return path to a model file: env override, MODEL_URL download, or models/."""
    env_path = os.environ.get("MODEL_PATH")
    if env_path and os.path.exists(env_path):
        return env_path

    url = os.environ.get("MODEL_URL")
    if url:
        downloaded = _download_weights(url)
        if downloaded:
            return downloaded

    if not os.path.isdir(MODELS_DIR):
        return None
    for name in sorted(os.listdir(MODELS_DIR)):
        if name.endswith((".keras", ".h5")):
            return os.path.join(MODELS_DIR, name)
    return None


def load_model():
    """Load the trained model if available; otherwise enable demo mode."""
    global _model, _demo_mode
    weights = _find_weights()
    if weights is None:
        _demo_mode = True
        return
    try:
        from tensorflow.keras.models import load_model as keras_load_model

        _model = keras_load_model(weights)
        _demo_mode = False
        print(f"[AlzDetect] Loaded model from {weights}")
    except Exception as exc:  # pragma: no cover - defensive
        print(f"[AlzDetect] Failed to load model ({exc}); falling back to demo mode.")
        _demo_mode = True


def is_demo_mode():
    return _demo_mode


def preprocess(image_bytes):
    """Bytes -> (1, 128, 128, 3) float32 array, matching the fuzzy notebook.

    IMPORTANT: the fuzzy MobileNetV2 model (the one deployed for live mode) was
    trained on RAW 0-255 pixel values — it defines an ImageDataGenerator with
    rescale=1/255 but never applies it in model.fit(). To get correct
    predictions we must replicate that exactly and NOT normalize here.
    If you retrain with proper 1/255 normalization, set NORMALIZE=True.
    """
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize((IMG_SIZE, IMG_SIZE))
    arr = np.asarray(img, dtype=np.float32)
    if os.environ.get("NORMALIZE", "0") == "1":
        arr = arr / 255.0
    return np.expand_dims(arr, axis=0)


def _demo_probs(image_bytes):
    """Deterministic pseudo-prediction from image content, so the UI is
    demonstrable without trained weights. Clearly flagged as demo."""
    digest = hashlib.sha256(image_bytes).digest()
    seed = int.from_bytes(digest[:8], "big")
    rng = np.random.default_rng(seed)
    logits = rng.normal(size=len(CLASS_NAMES))
    # Bias toward a confident-looking distribution.
    logits[rng.integers(len(CLASS_NAMES))] += 2.5
    exp = np.exp(logits - logits.max())
    return exp / exp.sum()


def predict(image_bytes):
    """Run inference. Returns dict with probabilities and prediction."""
    if _demo_mode or _model is None:
        probs = _demo_probs(image_bytes)
    else:
        x = preprocess(image_bytes)
        probs = _model.predict(x, verbose=0)[0]

    order = np.argsort(probs)[::-1]
    return {
        "demo": bool(_demo_mode or _model is None),
        "prediction": CLASS_NAMES[int(order[0])],
        "confidence": float(probs[int(order[0])]),
        "probabilities": [
            {"label": CLASS_NAMES[i], "probability": float(probs[i])}
            for i in order
        ],
    }


def gradcam(image_bytes):
    """Return a base64 PNG heatmap overlay showing model attention.

    Returns None in demo mode (no trained model to explain)."""
    if _demo_mode or _model is None:
        return None
    try:
        import base64

        import tensorflow as tf

        x = preprocess(image_bytes)
        grad_model = tf.keras.models.Model(
            _model.inputs,
            [_model.get_layer(_last_conv_layer).output, _model.output],
        )
        with tf.GradientTape() as tape:
            conv_out, preds = grad_model(x)
            class_idx = tf.argmax(preds[0])
            loss = preds[:, class_idx]
        grads = tape.gradient(loss, conv_out)
        pooled = tf.reduce_mean(grads, axis=(0, 1, 2))
        conv_out = conv_out[0]
        heatmap = conv_out @ pooled[..., None]
        heatmap = tf.squeeze(heatmap).numpy()
        heatmap = np.maximum(heatmap, 0)
        heatmap = heatmap / (heatmap.max() + 1e-8)

        # Upsample heatmap and overlay on the original image.
        heat_img = Image.fromarray(np.uint8(255 * heatmap)).resize(
            (IMG_SIZE, IMG_SIZE)
        )
        heat = np.asarray(heat_img, dtype=np.float32) / 255.0
        base = np.asarray(
            Image.open(io.BytesIO(image_bytes)).convert("RGB").resize(
                (IMG_SIZE, IMG_SIZE)
            ),
            dtype=np.float32,
        )
        # Simple red overlay.
        overlay = base.copy()
        overlay[..., 0] = np.minimum(255, base[..., 0] + heat * 180)
        blended = np.uint8(0.6 * base + 0.4 * overlay)
        out = Image.fromarray(blended)
        buf = io.BytesIO()
        out.save(buf, format="PNG")
        return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
    except Exception as exc:  # pragma: no cover - defensive
        print(f"[AlzDetect] Grad-CAM failed: {exc}")
        return None
