"""
TDSig disease predictor: lab Transformer (pooled) + ref/gen CNN features, concat + MLP head.
"""

import math
from typing import Dict

import torch
import torch.nn as nn


class FixedPositionalEncoding(nn.Module):
    def __init__(
        self,
        d_model: int,
        dropout: float = 0.1,
        max_len: int = 512,
    ) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.register_buffer("pe", self._build_pe(max_len, d_model), persistent=False)

    @staticmethod
    def _build_pe(max_len: int, d_model: int) -> torch.Tensor:
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_len, d_model, dtype=torch.float32)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)  # (1, max_len, d_model)

    def _ensure_length(self, seq_len: int) -> None:
        if seq_len <= self.pe.size(1):
            return
        self.pe = self._build_pe(seq_len, self.pe.size(-1)).to(self.pe.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(1)
        self._ensure_length(seq_len)
        positional = self.pe[:, :seq_len].to(device=x.device, dtype=x.dtype)
        return self.dropout(x + positional)


class LabTransformerEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int = 53,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 2,
        ff_dim: int = 256,
        out_dim: int = 128,
        dropout: float = 0.1,
        pooling: str = "mean",
        positional_max_len: int = 512,
    ) -> None:
        super().__init__()
        if d_model % nhead != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by nhead ({nhead}).")
        if pooling not in {"mean", "cls", "none"}:
            raise ValueError(f"Unsupported pooling mode: {pooling}.")

        self.input_dim = input_dim
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.ff_dim = ff_dim
        self.out_dim = out_dim
        self.pooling = pooling
        self.positional_max_len = positional_max_len

        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_encoder = FixedPositionalEncoding(
            d_model=d_model,
            dropout=dropout,
            max_len=positional_max_len,
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer=encoder_layer, num_layers=num_layers)
        self.cls_token = None
        if pooling == "cls":
            self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
            nn.init.normal_(self.cls_token, mean=0.0, std=0.02)

        self.output_proj = nn.Sequential(
            nn.Linear(d_model, out_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

    @staticmethod
    def _build_valid_time_mask(
        lab_lengths: torch.Tensor,
        seq_len: int,
        device: torch.device,
    ) -> torch.Tensor:
        time_index = torch.arange(seq_len, device=device).unsqueeze(0)
        return time_index < lab_lengths.unsqueeze(1)

    def forward(
        self,
        lab_test: torch.Tensor,
        lab_mask: torch.Tensor,
        lab_lengths: torch.Tensor,
    ) -> torch.Tensor:
        masked_lab = lab_test * lab_mask
        tokens = self.input_proj(masked_lab)  # (B, T, D)
        valid_mask = self._build_valid_time_mask(
            lab_lengths=lab_lengths,
            seq_len=lab_test.shape[1],
            device=lab_test.device,
        )  # (B, T), True means valid token.

        if self.pooling == "cls":
            batch_size = tokens.size(0)
            cls_tokens = self.cls_token.expand(batch_size, -1, -1)
            tokens = torch.cat([cls_tokens, tokens], dim=1)
            cls_mask = torch.ones(batch_size, 1, dtype=torch.bool, device=tokens.device)
            valid_mask = torch.cat([cls_mask, valid_mask], dim=1)

        tokens = self.pos_encoder(tokens)
        key_padding_mask = ~valid_mask  # Transformer expects True for padding positions.
        encoded = self.encoder(tokens, src_key_padding_mask=key_padding_mask)

        if self.pooling == "none":
            return self.output_proj(encoded)

        if self.pooling == "cls":
            pooled = encoded[:, 0]
        else:
            valid_mask_float = valid_mask.unsqueeze(-1).to(dtype=encoded.dtype)
            pooled = (encoded * valid_mask_float).sum(dim=1) / valid_mask_float.sum(dim=1).clamp(min=1.0)

        return self.output_proj(pooled)


class ImageEncoder(nn.Module):
    def __init__(self, out_dim: int = 128) -> None:
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.proj = nn.Linear(128, out_dim)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.backbone(images).flatten(1)
        return self.proj(features)


class TDSigDiseasePredictor(nn.Module):
    """Multimodal concat + MLP head (TDSig predictor)."""

    def __init__(
        self,
        lab_input_dim: int = 53,
        hidden_dim: int = 128,
        lab_d_model: int = 128,
        lab_nhead: int = 4,
        lab_num_layers: int = 2,
        lab_ff_dim: int = 256,
        lab_pooling: str = "mean",
        lab_positional_max_len: int = 512,
        image_feat_dim: int = 128,
        fusion_dim: int = 128,
        num_diseases: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.lab_encoder = LabTransformerEncoder(
            input_dim=lab_input_dim,
            d_model=lab_d_model,
            nhead=lab_nhead,
            num_layers=lab_num_layers,
            ff_dim=lab_ff_dim,
            out_dim=fusion_dim,
            dropout=dropout,
            pooling=lab_pooling,
            positional_max_len=lab_positional_max_len,
        )
        self.image_encoder = ImageEncoder(out_dim=image_feat_dim)

        self.ref_proj = nn.Linear(image_feat_dim, fusion_dim)
        self.gen_proj = nn.Linear(image_feat_dim, fusion_dim)

        self.fusion_head = nn.Sequential(
            nn.Linear(fusion_dim * 3, hidden_dim * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, num_diseases),
        )
        self._model_kwargs = {
            "lab_input_dim": lab_input_dim,
            "hidden_dim": hidden_dim,
            "lab_d_model": lab_d_model,
            "lab_nhead": lab_nhead,
            "lab_num_layers": lab_num_layers,
            "lab_ff_dim": lab_ff_dim,
            "lab_pooling": lab_pooling,
            "lab_positional_max_len": lab_positional_max_len,
            "image_feat_dim": image_feat_dim,
            "fusion_dim": fusion_dim,
            "num_diseases": num_diseases,
            "dropout": dropout,
        }

    def get_model_kwargs(self) -> Dict[str, object]:
        return dict(self._model_kwargs)

    def _pool_generated_features(
        self,
        gen_imgs: torch.Tensor,
        gen_img_mask: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, gen_count, channels, height, width = gen_imgs.shape
        flat_imgs = gen_imgs.view(batch_size * gen_count, channels, height, width)
        flat_features = self.image_encoder(flat_imgs).view(batch_size, gen_count, -1)

        valid_mask = gen_img_mask.unsqueeze(-1).float()
        pooled = (flat_features * valid_mask).sum(dim=1) / valid_mask.sum(dim=1).clamp(min=1.0)
        return pooled

    def forward(
        self,
        lab_test: torch.Tensor,
        lab_mask: torch.Tensor,
        lab_lengths: torch.Tensor,
        ref_img: torch.Tensor,
        gen_imgs: torch.Tensor,
        gen_img_mask: torch.Tensor,
        **kwargs: object,
    ) -> torch.Tensor:
        lab_feat = self.lab_encoder(lab_test, lab_mask, lab_lengths)

        ref_feat = self.image_encoder(ref_img)
        ref_feat = self.ref_proj(ref_feat)

        gen_feat = self._pool_generated_features(gen_imgs, gen_img_mask)
        gen_feat = self.gen_proj(gen_feat)

        fused_feat = torch.cat([lab_feat, ref_feat, gen_feat], dim=-1)
        logits = self.fusion_head(fused_feat)
        return logits
