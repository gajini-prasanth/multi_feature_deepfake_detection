import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from explainability.gradcam import MultiStreamGradCAM, overlay_heatmap
from models.ai_image_detector import AIImageDetector


def test_multistream_gradcam():
    model = AIImageDetector(streams=["residual", "frequency", "content"], fusion_type="cross_attention")
    cam = MultiStreamGradCAM(model)
    x = torch.randn(1, 3, 224, 224)
    res = cam.generate(x)
    cam.close()

    assert 0.0 <= res["probability"] <= 1.0
    assert set(res["cams"].keys()) == {"residual", "frequency", "content"}
    for c in res["cams"].values():
        assert c.shape == (224, 224)

    rgb = np.zeros((224, 224, 3), dtype=np.uint8)
    vis = overlay_heatmap(rgb, res["combined_cam"])
    assert vis.shape == (224, 224, 3)
