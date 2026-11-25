import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.augmentation import get_val_transforms
from data.dataset_loader import ImageForensicsDataset, collect_binary_folder_samples
from evaluation.metrics import compute_all_metrics, count_parameters, measure_inference_time
from models.ai_image_detector import AIImageDetector
from models.deepfake_detector import DeepfakeDetector


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Track A or Track B model")
    parser.add_argument("--track", type=str, required=True, choices=["a", "b"])
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--streams", nargs="+", default=["residual", "frequency", "content"])
    parser.add_argument("--fusion", type=str, default="cross_attention")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--output-json", type=str, default="outputs/eval_results.json")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cls = AIImageDetector if args.track == "a" else DeepfakeDetector
    model = cls(streams=args.streams, fusion_type=args.fusion).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    samples = collect_binary_folder_samples(args.data_dir)
    loader = DataLoader(
        ImageForensicsDataset(samples, transform=get_val_transforms(224), image_size=224),
        batch_size=args.batch_size,
        shuffle=False,
    )

    y_true, y_prob = [], []
    for x, y in loader:
        probs = torch.sigmoid(model(x.to(device))["logits"]).cpu().numpy()
        y_prob.extend(probs.tolist())
        y_true.extend(y.numpy().tolist())

    results = compute_all_metrics(y_true, y_prob, threshold=0.5)
    results["params"] = count_parameters(model)
    results["latency_ms"] = measure_inference_time(model)

    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
