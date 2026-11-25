import time
from typing import Dict, List

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def count_parameters(model: torch.nn.Module) -> int:
    """Return total number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


@torch.no_grad()
def measure_inference_time(
    model: torch.nn.Module,
    input_size: tuple[int, int, int, int] = (1, 3, 224, 224),
    runs: int = 20,
) -> float:
    """Measure average inference latency in milliseconds per image."""
    device = next(model.parameters()).device
    model.eval()
    x = torch.randn(*input_size, device=device)
    for _ in range(3):
        _ = model(x)
    if device.type == "cuda":
        torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(runs):
        _ = model(x)
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed_ms = (time.perf_counter() - start) * 1000.0 / (runs * input_size[0])
    return float(elapsed_ms)


def compute_all_metrics(
    y_true: np.ndarray | List[int],
    y_prob: np.ndarray | List[float],
    threshold: float = 0.5,
) -> Dict[str, object]:
    """Compute classification metrics at a fixed threshold (default 0.5)."""
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    y_pred = (y_prob >= threshold).astype(int)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    sensitivity = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0

    has_both = len(np.unique(y_true)) == 2
    roc_auc = float(roc_auc_score(y_true, y_prob)) if has_both else 0.0
    ap = float(average_precision_score(y_true, y_prob)) if has_both else 0.0

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": roc_auc,
        "average_precision": ap,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "confusion_matrix": cm.tolist(),
    }


def aggregate_seed_metrics(results: List[Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    """Aggregate scalar metrics across multiple random seeds as mean and std."""
    keys = [k for k, v in results[0].items() if isinstance(v, (int, float))]
    summary: Dict[str, Dict[str, float]] = {}
    for k in keys:
        vals = np.array([r[k] for r in results], dtype=float)
        summary[k] = {"mean": float(vals.mean()), "std": float(vals.std(ddof=0))}
    return summary
