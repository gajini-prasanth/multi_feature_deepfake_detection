from typing import Dict, List

import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss


def compute_ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    """Compute Expected Calibration Error (ECE)."""
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(y_true)
    if n == 0:
        return 0.0

    for i in range(n_bins):
        mask = (y_prob >= bins[i]) & (y_prob < bins[i + 1] if i < n_bins - 1 else y_prob <= bins[i + 1])
        if mask.any():
            acc = y_true[mask].mean()
            conf = y_prob[mask].mean()
            ece += (mask.sum() / n) * abs(acc - conf)
    return float(ece)


def compute_calibration_report(y_true: List[int] | np.ndarray, y_prob: List[float] | np.ndarray, n_bins: int = 10) -> Dict[str, object]:
    """Compute Brier score, ECE, and calibration curve bins."""
    y_true_arr = np.asarray(y_true, dtype=int)
    y_prob_arr = np.asarray(y_prob, dtype=float)
    brier = float(brier_score_loss(y_true_arr, y_prob_arr))
    ece = compute_ece(y_true_arr, y_prob_arr, n_bins=n_bins)
    frac_pos, mean_pred = calibration_curve(y_true_arr, y_prob_arr, n_bins=n_bins, strategy="uniform")
    return {
        "brier_score": brier,
        "ece": ece,
        "fraction_of_positives": frac_pos.tolist(),
        "mean_predicted_value": mean_pred.tolist(),
    }
