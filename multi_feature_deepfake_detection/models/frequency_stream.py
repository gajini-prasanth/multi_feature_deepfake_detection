import math
import torch
import torch.nn as nn


class FrequencyStream(nn.Module):
    """Stream B: 2D FFT (or block-wise DCT) log-magnitude spectrum -> CNN -> 256-d."""

    def __init__(self, feature_dim: int = 256, mode: str = "fft") -> None:
        super().__init__()
        if mode not in {"fft", "dct"}:
            raise ValueError(f"Unsupported frequency mode: {mode}")
        self.mode = mode

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 112x112
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 56x56
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 28x28
            nn.Conv2d(128, feature_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(feature_dim),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(4),  # 7x7
        )
        self.pool = nn.AdaptiveAvgPool2d(1)

    @staticmethod
    def _rgb_to_gray(x: torch.Tensor) -> torch.Tensor:
        r, g, b = x[:, 0:1], x[:, 1:2], x[:, 2:3]
        return 0.2989 * r + 0.5870 * g + 0.1140 * b

    def _compute_fft_spectrum(self, gray: torch.Tensor) -> torch.Tensor:
        fft = torch.fft.fft2(gray, norm="ortho")
        fft_shifted = torch.fft.fftshift(fft, dim=(-2, -1))
        mag = torch.log1p(torch.abs(fft_shifted))
        mean = mag.mean(dim=(-2, -1), keepdim=True)
        std = mag.std(dim=(-2, -1), keepdim=True).clamp(min=1e-6)
        return (mag - mean) / std

    def _compute_block_dct(self, gray: torch.Tensor, block_size: int = 8) -> torch.Tensor:
        b, c, h, w = gray.shape
        # Orthonormal 8x8 DCT-II basis
        n = torch.arange(block_size, device=gray.device, dtype=gray.dtype)
        k = n.unsqueeze(1)
        dct_mat = torch.cos(math.pi * (2 * n + 1) * k / (2 * block_size))
        dct_mat[0, :] *= 1.0 / math.sqrt(block_size)
        dct_mat[1:, :] *= math.sqrt(2.0 / block_size)

        blocks = gray.unfold(2, block_size, block_size).unfold(3, block_size, block_size)
        dct_blocks = torch.matmul(dct_mat, blocks)
        dct_blocks = torch.matmul(dct_blocks, dct_mat.t())
        out = dct_blocks.permute(0, 1, 2, 4, 3, 5).reshape(b, c, h, w)
        mag = torch.log1p(torch.abs(out))
        mean = mag.mean(dim=(-2, -1), keepdim=True)
        std = mag.std(dim=(-2, -1), keepdim=True).clamp(min=1e-6)
        return (mag - mean) / std

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        gray = self._rgb_to_gray(x)
        freq = self._compute_fft_spectrum(gray) if self.mode == "fft" else self._compute_block_dct(gray)
        feat_map = self.encoder(freq)  # [B, 256, 7, 7]
        feat_vec = self.pool(feat_map).flatten(1)  # [B, 256]
        return feat_map, feat_vec
