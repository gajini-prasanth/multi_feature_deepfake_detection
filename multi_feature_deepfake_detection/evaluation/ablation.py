from typing import Dict, List

import pandas as pd

from evaluation.metrics import count_parameters, measure_inference_time
from models.ai_image_detector import AIImageDetector


ABLATION_CONFIGS = [
    {"name": "Exp1_Residual_Only", "streams": ["residual"], "fusion": "concat"},
    {"name": "Exp1_Frequency_Only", "streams": ["frequency"], "fusion": "concat"},
    {"name": "Exp1_Content_Only", "streams": ["content"], "fusion": "concat"},
    {"name": "Exp2_Res_Freq", "streams": ["residual", "frequency"], "fusion": "concat"},
    {"name": "Exp2_Res_Content", "streams": ["residual", "content"], "fusion": "concat"},
    {"name": "Exp2_Freq_Content", "streams": ["frequency", "content"], "fusion": "concat"},
    {"name": "Exp3_ThreeStream_Concat", "streams": ["residual", "frequency", "content"], "fusion": "concat"},
    {"name": "Exp3_ThreeStream_CrossAttn", "streams": ["residual", "frequency", "content"], "fusion": "cross_attention"},
    {"name": "Exp3_ThreeStream_DualAttn", "streams": ["residual", "frequency", "content"], "fusion": "dual_attention"},
]


def profile_ablation_models() -> pd.DataFrame:
    """Profile parameter count and inference latency for all ablation configurations."""
    rows: List[Dict[str, object]] = []
    for cfg in ABLATION_CONFIGS:
        model = AIImageDetector(streams=cfg["streams"], fusion_type=cfg["fusion"])
        rows.append({
            "experiment": cfg["name"],
            "streams": "+".join(cfg["streams"]),
            "fusion": cfg["fusion"],
            "params": count_parameters(model),
            "latency_ms": round(measure_inference_time(model, runs=5), 3),
        })
    return pd.DataFrame(rows)
