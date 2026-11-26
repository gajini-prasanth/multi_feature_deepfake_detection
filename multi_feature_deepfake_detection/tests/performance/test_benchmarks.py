import json
import time
from pathlib import Path

import torch

from evaluation.metrics import count_parameters, measure_inference_time
from models.ai_image_detector import AIImageDetector


def test_performance_benchmarks():
    t0 = time.perf_counter()
    model = AIImageDetector(streams=["residual", "frequency", "content"], fusion_type="cross_attention")
    load_ms = (time.perf_counter() - t0) * 1000.0

    params = count_parameters(model)
    lat_b1 = measure_inference_time(model, input_size=(1, 3, 224, 224), runs=3)
    lat_b4 = measure_inference_time(model, input_size=(4, 3, 224, 224), runs=3)

    report = {
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "model_load_ms": round(load_ms, 2),
        "trainable_parameters": params,
        "latency_ms_per_image_b1": round(lat_b1, 2),
        "latency_ms_per_image_b4": round(lat_b4, 2),
    }
    out_path = Path(__file__).resolve().parents[2] / "reports" / "performance_report.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    assert params > 0
