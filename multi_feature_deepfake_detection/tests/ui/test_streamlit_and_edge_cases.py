import py_compile
from pathlib import Path

import numpy as np
import torch

from data.augmentation import get_val_transforms
from models.ai_image_detector import AIImageDetector


def test_streamlit_app_compiles():
    app_path = Path(__file__).resolve().parents[2] / "app" / "streamlit_app.py"
    py_compile.compile(str(app_path), doraise=True)


def test_edge_case_images():
    model = AIImageDetector(streams=["residual", "frequency", "content"], fusion_type="cross_attention")
    model.eval()
    tf = get_val_transforms(224)

    # Very small, very large, blank, and random noise images
    for img in [
        np.zeros((16, 16, 3), dtype=np.uint8),
        np.zeros((512, 512, 3), dtype=np.uint8),
        np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8),
    ]:
        x = tf(image=img)["image"].unsqueeze(0)
        with torch.no_grad():
            prob = torch.sigmoid(model(x)["logits"]).item()
        assert 0.0 <= prob <= 1.0
