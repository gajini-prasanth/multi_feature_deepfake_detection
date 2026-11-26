import argparse
import sys
from pathlib import Path

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.augmentation import get_train_transforms, get_val_transforms
from data.dataset_loader import collect_binary_folder_samples, create_dataloaders
from models.ai_image_detector import AIImageDetector
from training.losses import CombinedForensicLoss
from training.trainer import Trainer


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Track A: AI-Generated Image Detector")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--data-dir", type=str, required=True, help="Root folder with real/ and fake/")
    parser.add_argument("--streams", nargs="+", default=["residual", "frequency", "content"])
    parser.add_argument("--fusion", type=str, default="cross_attention", choices=["concat", "cross_attention", "dual_attention"])
    parser.add_argument("--use-contrastive", action="store_true")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--resume", type=str, default=None)
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    epochs = args.epochs or cfg["training"]["epochs"]
    batch_size = args.batch_size or cfg["training"]["batch_size"]
    img_size = cfg["project"]["image_size"]

    samples = collect_binary_folder_samples(args.data_dir)
    if len(samples) == 0:
        raise RuntimeError(f"No images found in {args.data_dir} (expected real/ and fake/ subfolders).")

    import random
    random.Random(cfg["project"]["seeds"][0]).shuffle(samples)
    split = int(0.8 * len(samples))
    train_samples, val_samples = samples[:split], samples[split:]

    train_loader, val_loader = create_dataloaders(
        train_samples,
        val_samples,
        get_train_transforms(img_size),
        get_val_transforms(img_size),
        batch_size=batch_size,
        num_workers=cfg["project"]["num_workers"],
        image_size=img_size,
    )

    model = AIImageDetector(
        streams=args.streams,
        fusion_type=args.fusion,
        feature_dim=cfg["model"]["feature_dim"],
        num_heads=cfg["model"]["num_heads"],
        fusion_layers=cfg["model"]["fusion_layers"],
        dropout=cfg["model"]["dropout"],
    )
    criterion = CombinedForensicLoss(
        use_contrastive=args.use_contrastive,
        contrastive_weight=cfg["model"]["contrastive_weight"],
        temperature=cfg["model"]["temperature"],
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["training"]["learning_rate"],
        weight_decay=cfg["training"]["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    trainer = Trainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        checkpoint_dir=cfg["training"]["checkpoint_dir"],
        use_amp=cfg["training"]["use_amp"],
        use_contrastive=args.use_contrastive,
    )
    start_epoch = trainer.load_checkpoint(args.resume) if args.resume else 1
    history = trainer.fit(train_loader, val_loader, epochs=epochs, start_epoch=start_epoch)
    best_src = Path(cfg["training"]["checkpoint_dir"]) / "best_model.pt"
    if best_src.exists():
        ckpt = torch.load(best_src, map_location="cpu")
        ckpt["config"] = cfg
        ckpt["class_labels"] = {0: "Real / Authentic", 1: "Fake / AI-Generated"}
        ckpt["image_size"] = img_size
        ckpt["val_metrics"] = history[-1] if history else {}
        torch.save(ckpt, Path(cfg["training"]["checkpoint_dir"]) / "best_ai_detector.pt")


if __name__ == "__main__":
    main()
