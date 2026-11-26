import torch

from models.ai_image_detector import AIImageDetector
from training.losses import CombinedForensicLoss


def test_full_model_forward_backward_and_checkpoint(tmp_path):
    model = AIImageDetector(streams=["residual", "frequency", "content"], fusion_type="cross_attention")
    criterion = CombinedForensicLoss(use_contrastive=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    x = torch.randn(2, 3, 224, 224)
    y = torch.tensor([0.0, 1.0])

    out = model(x, return_embedding=True)
    loss = criterion(out["logits"], y, out["embedding"])
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    ckpt_path = tmp_path / "model.pt"
    torch.save({"model_state_dict": model.state_dict()}, ckpt_path)

    reloaded = AIImageDetector(streams=["residual", "frequency", "content"], fusion_type="cross_attention")
    reloaded.load_state_dict(torch.load(ckpt_path, map_location="cpu")["model_state_dict"])
    reloaded.eval()
    with torch.no_grad():
        logits = reloaded(x)["logits"]
    assert logits.shape == (2,)
