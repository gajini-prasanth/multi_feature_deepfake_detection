import torch
import torch.nn as nn
import torch.nn.functional as F


class SupervisedContrastiveLoss(nn.Module):
    """Supervised Contrastive Loss (Khosla et al. / LASTED-inspired) for L2-normalized embeddings."""

    def __init__(self, temperature: float = 0.07) -> None:
        super().__init__()
        self.temperature = temperature

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        device = embeddings.device
        batch_size = embeddings.shape[0]
        if batch_size < 2:
            return torch.tensor(0.0, device=device)

        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device)
        logits_mask = torch.ones_like(mask) - torch.eye(batch_size, device=device)
        mask = mask * logits_mask

        sim = torch.matmul(embeddings, embeddings.T) / self.temperature
        sim_max, _ = torch.max(sim, dim=1, keepdim=True)
        logits = sim - sim_max.detach()

        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-8)

        pos_per_sample = mask.sum(1)
        valid = pos_per_sample > 0
        if not valid.any():
            return torch.tensor(0.0, device=device)

        mean_log_prob_pos = (mask[valid] * log_prob[valid]).sum(1) / pos_per_sample[valid]
        return -mean_log_prob_pos.mean()


class CombinedForensicLoss(nn.Module):
    """Binary Cross-Entropy + optional Supervised Contrastive Loss."""

    def __init__(
        self,
        use_contrastive: bool = False,
        contrastive_weight: float = 0.1,
        temperature: float = 0.07,
    ) -> None:
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.use_contrastive = use_contrastive
        self.contrastive_weight = contrastive_weight
        self.supcon = SupervisedContrastiveLoss(temperature=temperature) if use_contrastive else None

    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        embeddings: torch.Tensor | None = None,
    ) -> torch.Tensor:
        loss = self.bce(logits, labels)
        if self.use_contrastive and self.supcon is not None and embeddings is not None:
            loss = loss + self.contrastive_weight * self.supcon(embeddings, labels)
        return loss
