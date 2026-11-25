from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np


def load_rgb_image(image_path: str, target_size: int = 224) -> np.ndarray:
    """Load an image from disk as an RGB uint8 numpy array resized to target_size."""
    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if img.shape[0] != target_size or img.shape[1] != target_size:
        img = cv2.resize(img, (target_size, target_size), interpolation=cv2.INTER_AREA)
    return img


def verify_dataset_balance(samples: List[Tuple[str, int]]) -> Dict[str, int]:
    """Verify file existence and return class counts (0=real, 1=fake)."""
    counts = Counter()
    missing = 0
    for path, label in samples:
        if not Path(path).exists():
            missing += 1
        counts[int(label)] += 1
    return {
        "total": len(samples),
        "real_0": counts.get(0, 0),
        "fake_1": counts.get(1, 0),
        "missing_files": missing,
    }
