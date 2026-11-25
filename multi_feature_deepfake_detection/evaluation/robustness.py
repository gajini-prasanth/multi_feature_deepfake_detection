from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import torch
from torch.utils.data import DataLoader

from data.augmentation import get_degradation_transform
from data.dataset_loader import ImageForensicsDataset
from evaluation.metrics import compute_all_metrics


DEFAULT_DEGRADATIONS = [
    ("jpeg", 90),
    ("jpeg", 70),
    ("jpeg", 50),
    ("blur", 3),
    ("blur", 5),
    ("noise", 0.02),
    ("noise", 0.05),
    ("resize", 112),
    ("resize", 64),
]


@torch.no_grad()
def evaluate_robustness(
    model: torch.nn.Module,
    samples: List[Tuple[str, int]],
    degradations: List[Tuple[str, int | float]] = DEFAULT_DEGRADATIONS,
    batch_size: int = 32,
    image_size: int = 224,
    output_csv: str | None = None,
) -> pd.DataFrame:
    """Evaluate model robustness under JPEG, blur, noise, and resizing (Experiment 6)."""
    device = next(model.parameters()).device
    model.eval()
    rows: List[Dict[str, object]] = []

    for deg_type, severity in degradations:
        tf = get_degradation_transform(deg_type, severity, image_size=image_size)
        loader = DataLoader(
            ImageForensicsDataset(samples, transform=tf, image_size=image_size),
            batch_size=batch_size,
            shuffle=False,
        )
        y_true, y_prob = [], []
        for x, y in loader:
            probs = torch.sigmoid(model(x.to(device))["logits"]).cpu().numpy()
            y_prob.extend(probs.tolist())
            y_true.extend(y.numpy().tolist())

        metrics = compute_all_metrics(y_true, y_prob, threshold=0.5)
        metrics["degradation"] = deg_type
        metrics["severity"] = severity
        rows.append(metrics)

    df = pd.DataFrame(rows)
    if output_csv and not df.empty:
        Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_csv, index=False)
    return df
