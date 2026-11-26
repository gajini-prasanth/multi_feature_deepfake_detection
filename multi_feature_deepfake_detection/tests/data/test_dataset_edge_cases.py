import cv2
import numpy as np
import pytest

from data.augmentation import get_train_transforms, get_val_transforms
from data.dataset_loader import ImageForensicsDataset, create_dataloaders
from data.preprocessing import load_rgb_image, verify_dataset_balance
from data.video_processor import split_videos_without_leakage


def test_missing_and_corrupted_image_handling(tmp_path):
    missing = tmp_path / "does_not_exist.png"
    with pytest.raises(FileNotFoundError):
        load_rgb_image(str(missing))

    corrupt = tmp_path / "corrupt.png"
    corrupt.write_bytes(b"not a valid image")
    with pytest.raises(FileNotFoundError):
        load_rgb_image(str(corrupt))


def test_class_balance_and_dataloader(tmp_path):
    samples = []
    for i in range(4):
        p = tmp_path / f"img_{i}.png"
        cv2.imwrite(str(p), np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8))
        samples.append((str(p), i % 2))

    stats = verify_dataset_balance(samples)
    assert stats == {"total": 4, "real_0": 2, "fake_1": 2, "missing_files": 0}

    train_loader, val_loader = create_dataloaders(
        samples[:2], samples[2:], get_train_transforms(224), get_val_transforms(224), batch_size=2, num_workers=0
    )
    xb, yb = next(iter(train_loader))
    assert xb.shape == (2, 3, 224, 224)
    assert yb.shape == (2,)


def test_video_split_prevents_leakage():
    vids = [(f"video_{i}.mp4", i % 2) for i in range(30)]
    splits = split_videos_without_leakage(vids, train_ratio=0.7, val_ratio=0.15, seed=42)
    assert set(splits["train"]).isdisjoint(splits["val"])
    assert set(splits["train"]).isdisjoint(splits["test"])
    assert set(splits["val"]).isdisjoint(splits["test"])
