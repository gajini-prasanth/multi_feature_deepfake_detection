import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models.ai_image_detector import AIImageDetector
from training.losses import CombinedForensicLoss
from training.trainer import Trainer


def test_trainer_smoke_and_resume(tmp_path):
    x = torch.randn(4, 3, 224, 224)
    y = torch.tensor([0.0, 1.0, 0.0, 1.0])
    loader = DataLoader(TensorDataset(x, y), batch_size=2)

    model = AIImageDetector(streams=["residual", "frequency", "content"], fusion_type="cross_attention")
    criterion = CombinedForensicLoss(use_contrastive=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    trainer = Trainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        checkpoint_dir=str(tmp_path / "ckpt"),
        use_amp=False,
        use_contrastive=True,
    )
    history = trainer.fit(loader, loader, epochs=1)
    assert len(history) == 1
    assert (tmp_path / "ckpt" / "best_model.pt").exists()
    assert (tmp_path / "ckpt" / "last_checkpoint.pt").exists()

    next_epoch = trainer.load_checkpoint(tmp_path / "ckpt" / "last_checkpoint.pt")
    assert next_epoch == 2
