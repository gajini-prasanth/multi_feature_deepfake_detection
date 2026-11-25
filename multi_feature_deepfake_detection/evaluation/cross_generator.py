from pathlib import Path
from typing import Dict, List

import pandas as pd
import torch
from torch.utils.data import DataLoader

from data.augmentation import get_val_transforms
from data.dataset_loader import ImageForensicsDataset, collect_binary_folder_samples
from evaluation.metrics import compute_all_metrics


@torch.no_grad()
def evaluate_cross_generator(
    model: torch.nn.Module,
    genimage_root: str,
    generators: List[str],
    batch_size: int = 32,
    image_size: int = 224,
    output_csv: str | None = None,
) -> pd.DataFrame:
    """Evaluate a trained model across unseen GenImage generators (Experiment 5)."""
    device = next(model.parameters()).device
    model.eval()
    rows: List[Dict[str, object]] = []

    for gen in generators:
        gen_dir = Path(genimage_root) / gen
        samples = collect_binary_folder_samples(str(gen_dir))
        if not samples:
            continue
        loader = DataLoader(
            ImageForensicsDataset(samples, transform=get_val_transforms(image_size), image_size=image_size),
            batch_size=batch_size,
            shuffle=False,
        )
        y_true, y_prob = [], []
        for x, y in loader:
            probs = torch.sigmoid(model(x.to(device))["logits"]).cpu().numpy()
            y_prob.extend(probs.tolist())
            y_true.extend(y.numpy().tolist())

        metrics = compute_all_metrics(y_true, y_prob, threshold=0.5)
        metrics["generator"] = gen
        rows.append(metrics)

    df = pd.DataFrame(rows)
    if output_csv and not df.empty:
        Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_csv, index=False)
    return df
