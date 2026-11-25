import json
import sys
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

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
ckpt_path = st.sidebar.text_input("Checkpoint Path (optional)", value="")


@st.cache_resource
def load_model(track_name: str, checkpoint: str):
    cls = AIImageDetector if "Track A" in track_name else DeepfakeDetector
    model = cls(streams=["residual", "frequency", "content"], fusion_type="cross_attention")
    if checkpoint and Path(checkpoint).exists():
        state = torch.load(checkpoint, map_location="cpu")
        model.load_state_dict(state["model_state_dict"])
    model.eval()
    return model


model = load_model(task, ckpt_path)
uploaded = st.file_uploader("Upload an image (JPG/PNG)", type=["jpg", "jpeg", "png", "webp"])

if uploaded is not None:
    file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
    bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    rgb_224 = cv2.resize(rgb, (224, 224), interpolation=cv2.INTER_AREA)

    tensor = get_val_transforms(224)(image=rgb_224)["image"].unsqueeze(0)
    cam_gen = MultiStreamGradCAM(model)
    result = cam_gen.generate(tensor)
    cam_gen.close()

    prob = result["probability"]
    pred_label = "Fake / Manipulated" if prob >= 0.5 else "Real / Authentic"

    st.subheader(f"Prediction: {pred_label} (Confidence: {prob:.2%})")
    cols = st.columns(5)
    cols[0].image(rgb_224, caption="Original (224x224)", use_container_width=True)
    cols[1].image(overlay_heatmap(rgb_224, result["cams"]["residual"]), caption="Residual Grad-CAM", use_container_width=True)
    cols[2].image(overlay_heatmap(rgb_224, result["cams"]["frequency"]), caption="Frequency Grad-CAM", use_container_width=True)
    cols[3].image(overlay_heatmap(rgb_224, result["cams"]["content"]), caption="Content/Edge Grad-CAM", use_container_width=True)
    cols[4].image(overlay_heatmap(rgb_224, result["combined_cam"]), caption="Combined Grad-CAM", use_container_width=True)

    report = {
        "track": task,
        "prediction": pred_label,
        "fake_probability": round(prob, 4),
        "threshold": 0.5,
    }
    st.download_button(
        "Download Prediction Report (JSON)",
        data=json.dumps(report, indent=2),
        file_name="prediction_report.json",
        mime="application/json",
    )
