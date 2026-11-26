import json
import sys
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.augmentation import get_val_transforms
from explainability.gradcam import MultiStreamGradCAM, overlay_heatmap
from models.ai_image_detector import AIImageDetector
from models.deepfake_detector import DeepfakeDetector


st.set_page_config(page_title="Multi-Feature Deepfake & AI Image Detector", layout="wide")
st.title("Multi-Feature Deep Learning Forensic Detector")

task = st.sidebar.radio(
    "Select Detection Track",
    ["Track A: AI-Generated Image Detection", "Track B: Deepfake Face Detection"],
)
default_ckpt = (
    PROJECT_ROOT / "checkpoints" / "best_ai_detector.pt"
    if "Track A" in task
    else PROJECT_ROOT / "checkpoints" / "best_deepfake_detector.pt"
)
ckpt_path = st.sidebar.text_input("Checkpoint Path", value=str(default_ckpt))
threshold = st.sidebar.slider("Decision Threshold", 0.1, 0.9, 0.5, 0.05)
uncertain_margin = st.sidebar.slider("Uncertain Margin (+/-)", 0.0, 0.25, 0.10, 0.05)


@st.cache_resource
def load_model(track_name: str, checkpoint: str):
    ckpt_file = Path(checkpoint)
    if not ckpt_file.exists():
        return None, None
    cls = AIImageDetector if "Track A" in track_name else DeepfakeDetector
    model = cls(streams=["residual", "frequency", "content"], fusion_type="cross_attention")
    state = torch.load(ckpt_file, map_location="cpu")
    model.load_state_dict(state["model_state_dict"])
    model.eval()
    return model, state


model, ckpt_meta = load_model(task, ckpt_path)
if model is None:
    st.error(
        f"Trained checkpoint not found at `{ckpt_path}`. "
        "Inference is disabled so random/untrained weights are never used. "
        "Please train the model first using `training/train.py` or `training/train_deepfake.py`."
    )
    st.stop()

st.sidebar.success(f"Loaded checkpoint: {Path(ckpt_path).name}")
if isinstance(ckpt_meta, dict) and "epoch" in ckpt_meta:
    st.sidebar.info(f"Epoch: {ckpt_meta['epoch']} | Val Metrics: {ckpt_meta.get('val_metrics', {})}")

uploaded = st.file_uploader("Upload an image (JPG/PNG)", type=["jpg", "jpeg", "png", "webp"])

if uploaded is not None:
    file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
    bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if bgr is None:
        st.error("Could not decode the uploaded image.")
        st.stop()
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    rgb_224 = cv2.resize(rgb, (224, 224), interpolation=cv2.INTER_AREA)

    tensor = get_val_transforms(224)(image=rgb_224)["image"].unsqueeze(0)
    with torch.inference_mode():
        fake_prob = float(torch.sigmoid(model(tensor)["logits"]).item())

    cam_gen = MultiStreamGradCAM(model)
    result = cam_gen.generate(tensor)
    cam_gen.close()

    real_prob = 1.0 - fake_prob
    confidence = max(fake_prob, real_prob)
    if abs(fake_prob - threshold) < uncertain_margin:
        pred_label = "Uncertain"
    elif fake_prob >= threshold:
        pred_label = "Fake / Manipulated"
    else:
        pred_label = "Real / Authentic"

    st.subheader(f"Prediction: {pred_label} (Confidence: {confidence:.2%} | P(Fake)={fake_prob:.2%})")
    cols = st.columns(5)
    cols[0].image(rgb_224, caption="Original (224x224)", use_container_width=True)
    cols[1].image(overlay_heatmap(rgb_224, result["cams"]["residual"]), caption="Residual Grad-CAM", use_container_width=True)
    cols[2].image(overlay_heatmap(rgb_224, result["cams"]["frequency"]), caption="Frequency Grad-CAM", use_container_width=True)
    cols[3].image(overlay_heatmap(rgb_224, result["cams"]["content"]), caption="Content/Edge Grad-CAM", use_container_width=True)
    cols[4].image(overlay_heatmap(rgb_224, result["combined_cam"]), caption="Combined Grad-CAM", use_container_width=True)

    report = {
        "track": task,
        "checkpoint": str(ckpt_path),
        "prediction": pred_label,
        "confidence": round(confidence, 4),
        "fake_probability": round(fake_prob, 4),
        "threshold": threshold,
    }
    st.download_button(
        "Download Prediction Report (JSON)",
        data=json.dumps(report, indent=2),
        file_name="prediction_report.json",
        mime="application/json",
    )
