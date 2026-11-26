import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluation.calibration import compute_calibration_report


def test_calibration_report():
    y_true = [0, 0, 1, 1]
    y_prob = [0.1, 0.2, 0.8, 0.9]
    rep = compute_calibration_report(y_true, y_prob, n_bins=5)
    assert 0.0 <= rep["brier_score"] <= 1.0
    assert 0.0 <= rep["ece"] <= 1.0
    assert len(rep["fraction_of_positives"]) > 0
