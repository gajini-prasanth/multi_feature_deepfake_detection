import torch
import torch.nn as nn


class CrossAttentionBlock(nn.Module):
    """Single cross-attention + MLP block with residual connections and LayerNorm."""

    def __init__(self, dim: int = 256, num_heads: int = 8, dropout: float = 0.1) -> None:
        super().__init__()
        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.norm_mlp = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout),
        )

    def forward(self, query: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        q = self.norm_q(query)
        kv = self.norm_kv(context)
        attn_out, _ = self.attn(q, kv, kv, need_weights=False)
        x = query + attn_out
        x = x + self.mlp(self.norm_mlp(x))
        return x


class CrossMultiHeadAttentionFusion(nn.Module):
    """2-layer Cross Multi-Head Attention fusion across 2 or 3 stream token sequences."""

    def __init__(self, dim: int = 256, num_heads: int = 8, num_layers: int = 2, dropout: float = 0.1) -> None:
        super().__init__()
        self.layers = nn.ModuleList([
            nn.ModuleList([
                CrossAttentionBlock(dim=dim, num_heads=num_heads, dropout=dropout)
                for _ in range(3)
            ])
            for _ in range(num_layers)
        ])
        self.out_proj = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
        )

    def forward(self, feature_maps: list[torch.Tensor]) -> torch.Tensor:
        # Flatten [B, C, H, W] -> [B, H*W, C]
        tokens = [f.flatten(2).transpose(1, 2) for f in feature_maps]
        num_streams = len(tokens)

        for layer_blocks in self.layers:
            updated = []
            for i in range(num_streams):
                others = [tokens[j] for j in range(num_streams) if j != i]
                context = torch.cat(others, dim=1) if others else tokens[i]
                updated.append(layer_blocks[i](tokens[i], context))
            tokens = updated

        # Pool each stream's tokens and average across streams -> [B, C]
        pooled = torch.stack([t.mean(dim=1) for t in tokens], dim=0).mean(dim=0)
        return self.out_proj(pooled)
