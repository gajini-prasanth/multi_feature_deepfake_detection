import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluation.metrics import aggregate_seed_metrics, compute_all_metrics


def test_compute_all_metrics_and_aggregation():
    y_true = np.array([0, 0, 1, 1])
    y_prob = np.array([0.1, 0.4, 0.6, 0.9])
    m = compute_all_metrics(y_true, y_prob, threshold=0.5)
    assert m["accuracy"] == 1.0
    assert m["roc_auc"] == 1.0
    assert m["sensitivity"] == 1.0
    assert m["specificity"] == 1.0
    assert m["confusion_matrix"] == [[2, 0], [0, 2]]

    agg = aggregate_seed_metrics([m, m, m])
    assert agg["accuracy"]["mean"] == 1.0
    assert agg["accuracy"]["std"] == 0.0
