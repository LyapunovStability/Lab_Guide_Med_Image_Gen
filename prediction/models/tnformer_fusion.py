"""
TNformer-style temporal neighboring fusion (no missingness prompt).

Vectorized aggregation of imaging tokens onto lab time steps using multi-scale
time-difference kernels, then per-step fusion with lab tokens.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class FusionModule(nn.Module):
    """ELU-mix of two modalities with residual on the second branch + LayerNorm."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.lin_ct = nn.Linear(dim, dim)
        self.lin_ts = nn.Linear(dim, dim)
        self.lin_out = nn.Linear(dim, dim)
        self.act = nn.ELU()
        self.norm = nn.LayerNorm(dim)

    def forward(self, ct: torch.Tensor, ts: torch.Tensor) -> torch.Tensor:
        aux = self.act(self.lin_ct(ct) + self.lin_ts(ts))
        out = self.lin_out(aux) + ts
        return self.norm(out)


class TemporalNeighboringAggregator(nn.Module):
    """
    Map imaging events (reference + generated) onto each lab time step using
    multi-scale RBF-style weights over squared time differences (TNformer idea).
    """

    def __init__(
        self,
        image_feat_dim: int,
        hid_dim: int,
        num_scales: int = 4,
    ) -> None:
        super().__init__()
        if hid_dim % num_scales != 0:
            raise ValueError(f"hid_dim ({hid_dim}) must be divisible by num_scales ({num_scales}).")
        self.image_feat_dim = image_feat_dim
        self.hid_dim = hid_dim
        self.num_scales = num_scales
        self.d_sub = hid_dim // num_scales

        self.img_proj = nn.Linear(image_feat_dim, hid_dim)
        self.kernel_param = nn.Parameter(torch.randn(num_scales, self.d_sub))

    def forward(
        self,
        lab_times: torch.Tensor,
        img_times: torch.Tensor,
        img_feats: torch.Tensor,
        img_event_mask: torch.Tensor,
    ) -> torch.Tensor:
        b, k, _ = img_feats.shape
        t_lab = lab_times.shape[1]
        if lab_times.shape[0] != b or img_times.shape[0] != b or img_times.shape[1] != k:
            raise ValueError("Shape mismatch in TemporalNeighboringAggregator.")
        if img_event_mask.shape != (b, k):
            raise ValueError("img_event_mask must be (B, K).")

        x = self.img_proj(img_feats)  # (B, K, hid_dim)
        x = x.view(b, k, self.num_scales, self.d_sub)

        diff_sq = (lab_times.unsqueeze(2) - img_times.unsqueeze(1)).pow(2)
        mask_k = img_event_mask.unsqueeze(1).expand(b, t_lab, k)
        neg_inf = torch.finfo(diff_sq.dtype).min / 4
        diff_sq_masked = diff_sq.masked_fill(~mask_k, neg_inf)

        alpha = F.softplus(self.kernel_param)  # (num_scales, d_sub)

        outs = []
        for s in range(self.num_scales):
            scale = float(s + 1)
            logits = -scale * alpha[s].view(1, 1, 1, self.d_sub) * diff_sq_masked.unsqueeze(-1)
            weights = torch.softmax(logits, dim=2)
            weights = weights * mask_k.unsqueeze(-1).to(dtype=weights.dtype)
            w_sum = weights.sum(dim=2, keepdim=True).clamp(min=1e-8)
            weights = weights / w_sum

            x_s = x[:, :, s, :]  # (B, K, d_sub)
            contrib = torch.einsum("btkd,bkd->btd", weights, x_s)
            outs.append(contrib)

        return torch.cat(outs, dim=-1)


def masked_mean_pool(
    seq: torch.Tensor,
    lab_lengths: torch.Tensor,
) -> torch.Tensor:
    """seq: (B, T, D) — mean over valid steps only."""
    b, t_max, d = seq.shape
    idx = torch.arange(t_max, device=seq.device).unsqueeze(0)
    mask = idx < lab_lengths.unsqueeze(1)  # (B, T)
    mask_f = mask.unsqueeze(-1).to(dtype=seq.dtype)
    summed = (seq * mask_f).sum(dim=1)
    denom = mask_f.sum(dim=1).clamp(min=1.0)
    return summed / denom


class TNformerSequenceHead(nn.Module):
    """TransformerEncoder over fused sequence + masked mean + classifier."""

    def __init__(
        self,
        dim: int,
        nhead: int,
        num_layers: int,
        ff_dim: int,
        num_diseases: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if dim % nhead != 0:
            raise ValueError(f"dim ({dim}) must be divisible by nhead ({nhead}).")
        layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=nhead,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.pred = nn.Linear(dim, num_diseases)

    def forward(self, fused_seq: torch.Tensor, lab_lengths: torch.Tensor) -> torch.Tensor:
        b, t_max, _ = fused_seq.shape
        idx = torch.arange(t_max, device=fused_seq.device).unsqueeze(0)
        key_padding_mask = ~(idx < lab_lengths.unsqueeze(1))
        encoded = self.encoder(fused_seq, src_key_padding_mask=key_padding_mask)
        pooled = masked_mean_pool(encoded, lab_lengths)
        return self.pred(pooled)
