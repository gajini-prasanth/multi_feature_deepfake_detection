from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .content_stream import ContentStream
from .cross_attention import CrossMultiHeadAttentionFusion
from .frequency_stream import FrequencyStream
from .residual_stream import ResidualStream


class DualAttentionFusion(nn.Module):
    """MVSS-Net-inspired Spatial + Channel Dual Attention fusion."""

    def __init__(self, in_channels: int, out_dim: int = 256) -> None:
        super().__init__()
        self.channel_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // 4, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 4, in_channels, kernel_size=1),
            nn.Sigmoid(),
        )
        self.spatial_att = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False),
            nn.Sigmoid(),
        )
        self.proj = nn.Sequential(
            nn.Conv2d(in_channels, out_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )

    def forward(self, feature_maps: List[torch.Tensor]) -> torch.Tensor:
        x = torch.cat(feature_maps, dim=1)
        x = x * self.channel_att(x)
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = x * self.spatial_att(torch.cat([avg_out, max_out], dim=1))
        return self.proj(x).flatten(1)


class MultiFeatureFusionModel(nn.Module):
    """Configurable 1-, 2-, or 3-stream deepfake/AI-image detection model."""

    def __init__(
        self,
        streams: Optional[List[str]] = None,
        fusion_type: str = "cross_attention",
        feature_dim: int = 256,
        num_heads: int = 8,
        fusion_layers: int = 2,
        dropout: float = 0.3,
        freq_mode: str = "fft",
    ) -> None:
        super().__init__()
        self.streams_names = streams or ["residual", "frequency", "content"]
        self.fusion_type = fusion_type
        self.feature_dim = feature_dim

        self.stream_modules = nn.ModuleDict()
        if "residual" in self.streams_names:
            self.stream_modules["residual"] = ResidualStream(feature_dim=feature_dim)
        if "frequency" in self.streams_names:
            self.stream_modules["frequency"] = FrequencyStream(feature_dim=feature_dim, mode=freq_mode)
        if "content" in self.streams_names:
            self.stream_modules["content"] = ContentStream(feature_dim=feature_dim)

        n_streams = len(self.streams_names)
        if n_streams == 0:
            raise ValueError("At least one stream must be enabled.")

        if n_streams == 1 or fusion_type == "concat":
            fused_dim = feature_dim * n_streams
            self.fusion = None
        elif fusion_type == "cross_attention":
            fused_dim = feature_dim
            self.fusion = CrossMultiHeadAttentionFusion(
                dim=feature_dim, num_heads=num_heads, num_layers=fusion_layers, dropout=0.1
            )
        elif fusion_type == "dual_attention":
            fused_dim = feature_dim
            self.fusion = DualAttentionFusion(in_channels=feature_dim * n_streams, out_dim=feature_dim)
        else:
            raise ValueError(f"Unsupported fusion_type: {fusion_type}")

        self.classifier = nn.Sequential(
            nn.Linear(fused_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
        )
        self.embed_proj = nn.Linear(fused_dim, 128)

    def forward(self, x: torch.Tensor, return_embedding: bool = False) -> Dict[str, torch.Tensor]:
        fmaps, fvecs = [], []
        for name in self.streams_names:
            fmap, fvec = self.stream_modules[name](x)
            fmaps.append(fmap)
            fvecs.append(fvec)

        if self.fusion is None:
            fused = torch.cat(fvecs, dim=1)
        else:
            fused = self.fusion(fmaps)

        logits = self.classifier(fused).squeeze(1)
        out = {"logits": logits}
        if return_embedding:
            out["embedding"] = F.normalize(self.embed_proj(fused), dim=1)
        return out
