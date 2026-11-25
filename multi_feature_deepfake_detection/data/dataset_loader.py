from pathlib import Path
from typing import Callable, List, Optional, Tuple

import torch
from torch.utils.data import DataLoader, Dataset

from .preprocessing import load_rgb_image


class ImageForensicsDataset(Dataset):
    """Unified PyTorch Dataset for Track A (AI-generated) and Track B (deepfake faces)."""

    def __init__(
        self,
        samples: List[Tuple[str, int]],
        transform: Optional[Callable] = None,
        image_size: int = 224,
    ) -> None:
        self.samples = samples
        self.transform = transform
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img_path, label = self.samples[idx]
        image = load_rgb_image(img_path, target_size=self.image_size)
        if self.transform is not None:
            image = self.transform(image=image)["image"]
        else:
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        return image, torch.tensor(float(label), dtype=torch.float32)


def collect_binary_folder_samples(root_dir: str, real_dir: str = "real", fake_dir: str = "fake") -> List[Tuple[str, int]]:
    """Scan a directory with real/ and fake/ subfolders into (path, label) tuples."""
    root = Path(root_dir)
    exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    samples: List[Tuple[str, int]] = []

    for label, sub in [(0, real_dir), (1, fake_dir)]:
        folder = root / sub
        if not folder.exists():
            continue
        for p in sorted(folder.rglob("*")):
            if p.suffix.lower() in exts:
                samples.append((str(p), label))
    return samples


def create_dataloaders(
    train_samples: List[Tuple[str, int]],
    val_samples: List[Tuple[str, int]],
    train_transform: Optional[Callable] = None,
    val_transform: Optional[Callable] = None,
    batch_size: int = 32,
    num_workers: int = 4,
    image_size: int = 224,
) -> Tuple[DataLoader, DataLoader]:
    """Build train and validation DataLoaders."""
    train_ds = ImageForensicsDataset(train_samples, transform=train_transform, image_size=image_size)
    val_ds = ImageForensicsDataset(val_samples, transform=val_transform, image_size=image_size)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True
    )
    return train_loader, val_loader
