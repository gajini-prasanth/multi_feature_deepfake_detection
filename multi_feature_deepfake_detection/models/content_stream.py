import torch
import torch.nn as nn
import torch.nn.functional as F


class BayarConv2d(nn.Module):
    """Constrained 5x5 convolution inspired by Bayar & Stamm (center=-1, sum=0)."""

    def __init__(self, in_channels: int = 3, out_channels: int = 3, kernel_size: int = 5, padding: int = 2) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = padding
        self.kernel = nn.Parameter(
            torch.randn(out_channels, in_channels, kernel_size * kernel_size - 1) * 0.01
        )
        self.register_buffer("minus1", -torch.ones(out_channels, in_channels, 1))

    def _constrained_weight(self) -> torch.Tensor:
        w = self.kernel
        w_sum = w.sum(dim=2, keepdim=True)
        w_norm = w / (w_sum.abs() + 1e-6)
        # Re-scale so surrounding weights sum to +1.0
        w_norm = w_norm / (w_norm.sum(dim=2, keepdim=True) + 1e-6)
        ctr = (self.kernel_size * self.kernel_size) // 2
        full = torch.cat([w_norm[:, :, :ctr], self.minus1, w_norm[:, :, ctr:]], dim=2)
        return full.view(self.out_channels, self.in_channels, self.kernel_size, self.kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.conv2d(x, self._constrained_weight(), bias=None, stride=1, padding=self.padding)


class SobelEdgeExtractor(nn.Module):
    """Fixed Sobel X/Y edge magnitude extractor."""

    def __init__(self) -> None:
        super().__init__()
        sx = torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=torch.float32)
        sy = torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=torch.float32)
        self.register_buffer("sobel_x", sx.view(1, 1, 3, 3).repeat(3, 1, 1, 1))
        self.register_buffer("sobel_y", sy.view(1, 1, 3, 3).repeat(3, 1, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gx = F.conv2d(x, self.sobel_x, padding=1, groups=3)
        gy = F.conv2d(x, self.sobel_y, padding=1, groups=3)
        return torch.sqrt(gx * gx + gy * gy + 1e-6)


class ContentStream(nn.Module):
    """Stream C: RGB + BayarConv + Sobel Edge -> CNN -> 256-d feature map & vector."""

    def __init__(self, feature_dim: int = 256) -> None:
        super().__init__()
        self.bayar = BayarConv2d(in_channels=3, out_channels=3, kernel_size=5, padding=2)
        self.sobel = SobelEdgeExtractor()

        # 3 (RGB) + 3 (Bayar) + 3 (Sobel) = 9 input channels
        self.encoder = nn.Sequential(
            nn.Conv2d(9, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 112x112
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 56x56
            nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 28x28
            nn.Conv2d(256, feature_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(feature_dim),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(4),  # 7x7
        )
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        bayar_feat = self.bayar(x)
        edge_feat = self.sobel(x)
        combined = torch.cat([x, bayar_feat, edge_feat], dim=1)  # [B, 9, 224, 224]
        feat_map = self.encoder(combined)  # [B, 256, 7, 7]
        feat_vec = self.pool(feat_map).flatten(1)  # [B, 256]
        return feat_map, feat_vec
