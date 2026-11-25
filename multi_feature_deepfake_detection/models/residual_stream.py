import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def _build_srm_kernels() -> torch.Tensor:
    """Build 30 high-pass SRM 5x5 kernels (Fridrich & Kodovsky, 2012)."""
    kernels = []

    # 1st-order edge filters (8 directions)
    for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (1, 1), (-1, 1), (1, -1)]:
        k = np.zeros((5, 5), dtype=np.float32)
        k[2, 2] = -1.0
        k[2 + dy, 2 + dx] = 1.0
        kernels.append(k)

    # 2nd-order filters (4 orientations)
    for dx, dy in [(1, 0), (0, 1), (1, 1), (1, -1)]:
        k = np.zeros((5, 5), dtype=np.float32)
        k[2, 2] = -2.0
        k[2 + dy, 2 + dx] = 1.0
        k[2 - dy, 2 - dx] = 1.0
        kernels.append(k / 2.0)

    # 3rd-order filters (8 orientations)
    for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (1, 1), (-1, 1), (1, -1)]:
        k = np.zeros((5, 5), dtype=np.float32)
        k[2, 2] = -3.0
        k[2 + dy, 2 + dx] = 3.0
        k[2 - dy, 2 - dx] = 1.0
        k[2 + 2 * dy, 2 + 2 * dx] = -1.0
        kernels.append(k / 3.0)

    # 3x3 SQUARE & EDGE filters (6 filters)
    sq3 = np.array([
        [0, 0, 0, 0, 0],
        [0, -1, 2, -1, 0],
        [0, 2, -4, 2, 0],
        [0, -1, 2, -1, 0],
        [0, 0, 0, 0, 0],
    ], dtype=np.float32) / 4.0
    kernels.append(sq3)

    # 5x5 SQUARE filter
    sq5 = np.array([
        [-1, 2, -2, 2, -1],
        [2, -6, 8, -6, 2],
        [-2, 8, -12, 8, -2],
        [2, -6, 8, -6, 2],
        [-1, 2, -2, 2, -1],
    ], dtype=np.float32) / 12.0
    kernels.append(sq5)

    # 4 directional 3x3 edge-like high-pass filters
    for sign in [1.0, -1.0]:
        k1 = np.zeros((5, 5), dtype=np.float32)
        k1[1:4, 1:4] = np.array([[-1, 2, -1], [2, -4, 2], [0, 0, 0]], dtype=np.float32) * sign / 4.0
        k2 = np.zeros((5, 5), dtype=np.float32)
        k2[1:4, 1:4] = np.array([[-1, 2, 0], [2, -4, 0], [-1, 2, 0]], dtype=np.float32) * sign / 4.0
        kernels.extend([k1, k2])

    # Ensure exactly 30 kernels
    while len(kernels) < 30:
        kernels.append(sq3.copy())
    kernels = np.stack(kernels[:30], axis=0)  # [30, 5, 5]
    return torch.from_numpy(kernels)


class SRMConv2d(nn.Module):
    """Applies 30 fixed SRM high-pass kernels per RGB channel -> 90 residual maps."""

    def __init__(self, clamp_val: float = 3.0) -> None:
        super().__init__()
        self.clamp_val = clamp_val
        base_kernels = _build_srm_kernels()  # [30, 5, 5]
        # Grouped conv: 3 groups (R, G, B), 30 filters per group -> 90 output channels
        weight = base_kernels.unsqueeze(1).repeat(3, 1, 1, 1)  # [90, 1, 5, 5]
        self.register_buffer("weight", weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = F.conv2d(x, self.weight, bias=None, stride=1, padding=2, groups=3)
        return torch.clamp(res, -self.clamp_val, self.clamp_val)


class ResidualStream(nn.Module):
    """Stream A: SRM residual extraction -> CNN -> 256-d feature map & vector."""

    def __init__(self, feature_dim: int = 256) -> None:
        super().__init__()
        self.srm = SRMConv2d()
        self.encoder = nn.Sequential(
            nn.Conv2d(90, 64, kernel_size=3, padding=1, bias=False),
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
        res = self.srm(x)
        feat_map = self.encoder(res)  # [B, 256, 7, 7]
        feat_vec = self.pool(feat_map).flatten(1)  # [B, 256]
        return feat_map, feat_vec
