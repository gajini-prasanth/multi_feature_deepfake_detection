import random
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np


def split_videos_without_leakage(
    video_items: List[Tuple[str, int]],
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> Dict[str, List[Tuple[str, int]]]:
    """Split video-level items into train/val/test sets so frames never leak across splits."""
    rng = random.Random(seed)
    items = list(video_items)
    rng.shuffle(items)

    n = len(items)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    return {
        "train": items[:n_train],
        "val": items[n_train : n_train + n_val],
        "test": items[n_train + n_val :],
    }


def extract_frames_and_faces(
    video_path: str,
    output_dir: str,
    num_frames: int = 16,
    target_size: int = 224,
    margin: float = 0.25,
) -> List[str]:
    """Extract uniformly sampled frames from a video and crop the largest detected face."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        cap.release()
        return []

    indices = np.linspace(0, total_frames - 1, min(num_frames, total_frames), dtype=int)
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    saved_paths: List[str] = []

    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if not ret or frame is None:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40))

        h, w = frame.shape[:2]
        if len(faces) > 0:
            x, y, fw, fh = max(faces, key=lambda b: b[2] * b[3])
            mx, my = int(fw * margin), int(fh * margin)
            x1 = max(0, x - mx)
            y1 = max(0, y - my)
            x2 = min(w, x + fw + mx)
            y2 = min(h, y + fh + my)
            crop = frame[y1:y2, x1:x2]
        else:
            side = min(h, w)
            y1 = (h - side) // 2
            x1 = (w - side) // 2
            crop = frame[y1 : y1 + side, x1 : x1 + side]

        resized = cv2.resize(crop, (target_size, target_size), interpolation=cv2.INTER_AREA)
        frame_file = out_path / f"frame_{idx:05d}.png"
        cv2.imwrite(str(frame_file), resized)
        saved_paths.append(str(frame_file))

    cap.release()
    return saved_paths
