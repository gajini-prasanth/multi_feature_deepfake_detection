import json
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm


class Trainer:
    """Reusable trainer with AMP, checkpointing, and resume support."""

    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler._LRScheduler | None = None,
        checkpoint_dir: str = "checkpoints",
        use_amp: bool = True,
        use_contrastive: bool = False,
    ) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.use_amp = use_amp and self.device.type == "cuda"
        self.use_contrastive = use_contrastive
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)
        self.best_val_loss = float("inf")
        self.history: List[Dict[str, float]] = []

    def _run_epoch(self, loader: DataLoader, train: bool) -> Dict[str, float]:
        self.model.train(train)
        total_loss = 0.0
        correct = 0
        total = 0

        for images, labels in tqdm(loader, leave=False, desc="Train" if train else "Val"):
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            if train:
                self.optimizer.zero_grad(set_to_none=True)

            with torch.set_grad_enabled(train):
                with torch.amp.autocast(device_type=self.device.type, enabled=self.use_amp):
                    out = self.model(images, return_embedding=self.use_contrastive)
                    loss = self.criterion(
                        out["logits"], labels, out.get("embedding")
                    )

                if train:
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()

            preds = (torch.sigmoid(out["logits"]) >= 0.5).float()
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            total_loss += loss.item() * labels.size(0)

        return {
            "loss": total_loss / max(total, 1),
            "acc": correct / max(total, 1),
        }

    def save_checkpoint(self, epoch: int, filename: str) -> Path:
        path = self.checkpoint_dir / filename
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler else None,
                "best_val_loss": self.best_val_loss,
            },
            path,
        )
        return path

    def load_checkpoint(self, path: str | Path) -> int:
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if self.scheduler and ckpt.get("scheduler_state_dict"):
            self.scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        self.best_val_loss = ckpt.get("best_val_loss", float("inf"))
        return int(ckpt["epoch"]) + 1

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int,
        start_epoch: int = 1,
        patience: int = 7,
    ) -> List[Dict[str, float]]:
        no_improve = 0
        for epoch in range(start_epoch, epochs + 1):
            train_metrics = self._run_epoch(train_loader, train=True)
            val_metrics = self._run_epoch(val_loader, train=False)

            if self.scheduler is not None:
                self.scheduler.step()

            record = {
                "epoch": epoch,
                "train_loss": train_metrics["loss"],
                "train_acc": train_metrics["acc"],
                "val_loss": val_metrics["loss"],
                "val_acc": val_metrics["acc"],
            }
            self.history.append(record)

            self.save_checkpoint(epoch, "last_checkpoint.pt")
            if val_metrics["loss"] < self.best_val_loss:
                self.best_val_loss = val_metrics["loss"]
                self.save_checkpoint(epoch, "best_model.pt")
                no_improve = 0
            else:
                no_improve += 1
                if patience > 0 and no_improve >= patience:
                    break

        with open(self.checkpoint_dir / "history.json", "w", encoding="utf-8") as f:
            json.dump(self.history, f, indent=2)

        return self.history
