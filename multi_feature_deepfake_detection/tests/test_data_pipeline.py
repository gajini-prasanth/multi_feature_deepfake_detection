import sys
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.augmentation import get_degradation_transform, get_train_transforms, get_val_transforms
from data.dataset_loader import ImageForensicsDataset, create_dataloaders
from data.preprocessing import verify_dataset_balance
from data.video_processor import split_videos_without_leakage


def test_dataset_and_transforms(tmp_path):
    real_img = tmp_path / "real.png"
    fake_img = tmp_path / "fake.png"
    dummy = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    cv2.imwrite(str(real_img), dummy)
    cv2.imwrite(str(fake_img), dummy)

    samples = [(str(real_img), 0), (str(fake_img), 1)]
    stats = verify_dataset_balance(samples)
    assert stats == {"total": 2, "real_0": 1, "fake_1": 1, "missing_files": 0}

    ds = ImageForensicsDataset(samples, transform=get_train_transforms(224))
    img_t, label_t = ds[0]
    assert img_t.shape == (3, 224, 224)
    assert label_t.dtype == torch.float32

    train_loader, val_loader = create_dataloaders(
        samples, samples, get_train_transforms(224), get_val_transforms(224), batch_size=2, num_workers=0
    )
    xb, yb = next(iter(train_loader))
    assert xb.shape == (2, 3, 224, 224)
    assert yb.shape == (2,)

    deg = get_degradation_transform("jpeg", 70, 224)
    out = deg(image=dummy)["image"]
    assert out.shape == (3, 224, 224)


def test_video_split_no_leakage():
    videos = [(f"vid_{i}.mp4", i % 2) for i in range(20)]
    splits = split_videos_without_leakage(videos, train_ratio=0.7, val_ratio=0.15, seed=42)
    train_set = set(splits["train"])
    val_set = set(splits["val"])
    test_set = set(splits["test"])
    assert len(train_set & val_set) == 0
    assert len(train_set & test_set) == 0
    assert len(val_set & test_set) == 0
    assert len(train_set) + len(val_set) + len(test_set) == 20
