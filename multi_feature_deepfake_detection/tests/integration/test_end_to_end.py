import cv2
import numpy as np
import torch

from data.augmentation import get_val_transforms
from data.preprocessing import load_rgb_image
from explainability.gradcam import MultiStreamGradCAM, overlay_heatmap
from models.ai_image_detector import AIImageDetector
from models.deepfake_detector import DeepfakeDetector


def test_end_to_end_track_a_and_track_b(tmp_path):
    img_path = tmp_path / "sample.png"
    cv2.imwrite(str(img_path), np.random.randint(0, 255, (300, 300, 3), dtype=np.uint8))

    rgb = load_rgb_image(str(img_path), target_size=224)
    tensor = get_val_transforms(224)(image=rgb)["image"].unsqueeze(0)

    for cls in (AIImageDetector, DeepfakeDetector):
        model = cls(streams=["residual", "frequency", "content"], fusion_type="cross_attention")
        cam = MultiStreamGradCAM(model)
        res = cam.generate(tensor)
        cam.close()
        assert 0.0 <= res["probability"] <= 1.0
        vis = overlay_heatmap(rgb, res["combined_cam"])
        assert vis.shape == (224, 224, 3)
