import pytest
import torch
from hypothesis import given, settings
from hypothesis import strategies as st

from explainability.gradcam import MultiStreamGradCAM
from models.ai_image_detector import AIImageDetector
from models.content_stream import BayarConv2d, ContentStream, SobelEdgeExtractor
from models.cross_attention import CrossMultiHeadAttentionFusion
from models.frequency_stream import FrequencyStream
from models.residual_stream import ResidualStream, SRMConv2d


def test_srm_kernels_and_residual_channels(dummy_batch):
    srm = SRMConv2d()
    assert srm.weight.shape == (90, 1, 5, 5)
    res = srm(dummy_batch)
    assert res.shape == (2, 90, 224, 224)
    assert not torch.isnan(res).any()
    assert not torch.isinf(res).any()

    stream = ResidualStream(feature_dim=256)
    fmap, fvec = stream(dummy_batch)
    assert fmap.shape == (2, 256, 7, 7)
    assert fvec.shape == (2, 256)


def test_frequency_stream_fft_and_dct_stability(dummy_batch):
    for mode in ("fft", "dct"):
        stream = FrequencyStream(feature_dim=256, mode=mode)
        fmap, fvec = stream(dummy_batch)
        assert fmap.shape == (2, 256, 7, 7)
        assert fvec.shape == (2, 256)
        assert not torch.isnan(fvec).any()
        assert not torch.isinf(fvec).any()


def test_content_stream_bayar_and_sobel(dummy_batch):
    bayar = BayarConv2d()
    b_out = bayar(dummy_batch)
    assert b_out.shape == (2, 3, 224, 224)

    sobel = SobelEdgeExtractor()
    s_out = sobel(dummy_batch)
    assert s_out.shape == (2, 3, 224, 224)
    assert (s_out >= 0).all()

    stream = ContentStream(feature_dim=256)
    fmap, fvec = stream(dummy_batch)
    assert fmap.shape == (2, 256, 7, 7)
    assert fvec.shape == (2, 256)


@given(batch_size=st.integers(min_value=1, max_value=4))
@settings(max_examples=3, deadline=None)
def test_cross_attention_various_batch_sizes(batch_size):
    fusion = CrossMultiHeadAttentionFusion(dim=256, num_heads=8, num_layers=2)
    fmaps = [torch.randn(batch_size, 256, 7, 7, requires_grad=True) for _ in range(3)]
    out = fusion(fmaps)
    assert out.shape == (batch_size, 256)
    assert not torch.isnan(out).any()
    out.sum().backward()
    assert all(f.grad is not None for f in fmaps)


def test_classification_probability_range(dummy_batch):
    model = AIImageDetector(streams=["residual", "frequency", "content"], fusion_type="cross_attention")
    model.eval()
    with torch.no_grad():
        probs = torch.sigmoid(model(dummy_batch)["logits"])
    assert probs.shape == (2,)
    assert ((probs >= 0.0) & (probs <= 1.0)).all()


def test_gradcam_unit(dummy_single):
    model = AIImageDetector(streams=["residual", "frequency", "content"], fusion_type="cross_attention")
    cam = MultiStreamGradCAM(model)
    res = cam.generate(dummy_single)
    cam.close()
    assert set(res["cams"].keys()) == {"residual", "frequency", "content"}
    assert res["combined_cam"].shape == (224, 224)
