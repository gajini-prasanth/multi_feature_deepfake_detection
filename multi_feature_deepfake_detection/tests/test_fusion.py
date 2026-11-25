import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models.ai_image_detector import AIImageDetector
from models.deepfake_detector import DeepfakeDetector


def test_single_and_two_stream_configs():
    x = torch.randn(2, 3, 224, 224)
    for streams in [
        ["residual"],
        ["frequency"],
        ["content"],
        ["residual", "frequency"],
        ["residual", "content"],
        ["frequency", "content"],
    ]:
        model = AIImageDetector(streams=streams, fusion_type="concat")
        out = model(x, return_embedding=True)
        assert out["logits"].shape == (2,)
        assert out["embedding"].shape == (2, 128)


def test_three_stream_fusion_strategies_and_gradients():
    for fusion_type in ("concat", "cross_attention", "dual_attention"):
        x = torch.randn(2, 3, 224, 224, requires_grad=True)
        model = DeepfakeDetector(
            streams=["residual", "frequency", "content"],
            fusion_type=fusion_type,
        )
        out = model(x, return_embedding=True)
        assert out["logits"].shape == (2,)
        assert out["embedding"].shape == (2, 128)
        out["logits"].sum().backward()
        assert x.grad is not None
