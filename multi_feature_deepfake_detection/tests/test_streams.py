import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models.residual_stream import ResidualStream
from models.frequency_stream import FrequencyStream
from models.content_stream import ContentStream


def test_residual_stream():
    x = torch.randn(2, 3, 224, 224, requires_grad=True)
    model = ResidualStream(feature_dim=256)
    fmap, fvec = model(x)
    assert fmap.shape == (2, 256, 7, 7)
    assert fvec.shape == (2, 256)
    fvec.sum().backward()
    assert x.grad is not None


def test_frequency_stream_fft_and_dct():
    for mode in ("fft", "dct"):
        x = torch.randn(2, 3, 224, 224, requires_grad=True)
        model = FrequencyStream(feature_dim=256, mode=mode)
        fmap, fvec = model(x)
        assert fmap.shape == (2, 256, 7, 7)
        assert fvec.shape == (2, 256)
        fvec.sum().backward()
        assert x.grad is not None


def test_content_stream():
    x = torch.randn(2, 3, 224, 224, requires_grad=True)
    model = ContentStream(feature_dim=256)
    fmap, fvec = model(x)
    assert fmap.shape == (2, 256, 7, 7)
    assert fvec.shape == (2, 256)
    fvec.sum().backward()
    assert x.grad is not None
