"""
TNformer-style disease predictor: temporal neighboring of imaging onto lab sequence,
fusion per timestep, then TransformerEncoder + head (no missing-modality prompt).
"""

from __future__ import annotations

from typing import Any, Dict

import torch
import torch.nn as nn

from prediction.models.tdsig import ImageEncoder, LabTransformerEncoder
from prediction.models.tnformer_fusion import (
    FusionModule,
    TemporalNeighboringAggregator,
    TNformerSequenceHead,
)


class TnformerDiseasePredictor(nn.Module):
    def __init__(
        self,
        lab_input_dim: int = 53,
        hidden_dim: int = 128,
        lab_d_model: int = 128,
        lab_nhead: int = 4,
        lab_num_layers: int = 2,
        lab_ff_dim: int = 256,
        lab_positional_max_len: int = 512,
        image_feat_dim: int = 128,
        fusion_dim: int = 128,
        num_diseases: int = 4,
        dropout: float = 0.1,
        tn_num_scales: int = 4,
        tn_fusion_nhead: int = 4,
        tn_fusion_layers: int = 2,
        tn_fusion_ff_dim: int | None = None,
    ) -> None:
        super().__init__()
        if tn_fusion_ff_dim is None:
            tn_fusion_ff_dim = fusion_dim * 2

        self.lab_encoder = LabTransformerEncoder(
            input_dim=lab_input_dim,
            d_model=lab_d_model,
            nhead=lab_nhead,
            num_layers=lab_num_layers,
            ff_dim=lab_ff_dim,
            out_dim=fusion_dim,
            dropout=dropout,
            pooling="none",
            positional_max_len=lab_positional_max_len,
        )
        self.image_encoder = ImageEncoder(out_dim=image_feat_dim)

        self.tn_agg = TemporalNeighboringAggregator(
            image_feat_dim=image_feat_dim,
            hid_dim=fusion_dim,
            num_scales=tn_num_scales,
        )
        self.fusion = FusionModule(fusion_dim)
        self.seq_head = TNformerSequenceHead(
            dim=fusion_dim,
            nhead=tn_fusion_nhead,
            num_layers=tn_fusion_layers,
            ff_dim=tn_fusion_ff_dim,
            num_diseases=num_diseases,
            dropout=dropout,
        )

        self._model_kwargs: Dict[str, Any] = {
            "lab_input_dim": lab_input_dim,
            "hidden_dim": hidden_dim,
            "lab_d_model": lab_d_model,
            "lab_nhead": lab_nhead,
            "lab_num_layers": lab_num_layers,
            "lab_ff_dim": lab_ff_dim,
            "lab_positional_max_len": lab_positional_max_len,
            "image_feat_dim": image_feat_dim,
            "fusion_dim": fusion_dim,
            "num_diseases": num_diseases,
            "dropout": dropout,
            "tn_num_scales": tn_num_scales,
            "tn_fusion_nhead": tn_fusion_nhead,
            "tn_fusion_layers": tn_fusion_layers,
            "tn_fusion_ff_dim": tn_fusion_ff_dim,
        }

    def get_model_kwargs(self) -> Dict[str, Any]:
        return dict(self._model_kwargs)

    def _build_image_events(
        self,
        ref_img: torch.Tensor,
        gen_imgs: torch.Tensor,
        gen_img_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Stack ref + gen encoder features and build per-event validity mask."""
        b = ref_img.shape[0]
        ref_feat = self.image_encoder(ref_img)
        ref_stacked = ref_feat.unsqueeze(1)  # (B, 1, F)

        _, g_max, c, h, w = gen_imgs.shape
        if g_max == 0:
            return ref_stacked, torch.ones(b, 1, dtype=torch.bool, device=ref_img.device)

        flat = gen_imgs.view(b * g_max, c, h, w)
        gen_feat = self.image_encoder(flat).view(b, g_max, -1)
        feats = torch.cat([ref_stacked, gen_feat], dim=1)
        mask_ref = torch.ones(b, 1, dtype=torch.bool, device=ref_img.device)
        img_event_mask = torch.cat([mask_ref, gen_img_mask], dim=1)
        return feats, img_event_mask

    def forward(
        self,
        lab_test: torch.Tensor,
        lab_mask: torch.Tensor,
        lab_lengths: torch.Tensor,
        lab_times: torch.Tensor,
        ref_img: torch.Tensor,
        gen_imgs: torch.Tensor,
        gen_img_mask: torch.Tensor,
        ref_time: torch.Tensor,
        gen_times: torch.Tensor,
    ) -> torch.Tensor:
        lab_seq = self.lab_encoder(lab_test, lab_mask, lab_lengths)  # (B, T, D)

        img_feats, event_mask = self._build_image_events(ref_img, gen_imgs, gen_img_mask)
        _b, k, _ = img_feats.shape

        ref_t = ref_time.unsqueeze(1)  # (B, 1)
        if gen_imgs.shape[1] == 0:
            img_times = ref_t
        else:
            img_times = torch.cat([ref_t, gen_times], dim=1)
        if img_times.shape[1] != k:
            raise ValueError("img_times length must match stacked image features.")

        aligned = self.tn_agg(
            lab_times=lab_times,
            img_times=img_times,
            img_feats=img_feats,
            img_event_mask=event_mask,
        )
        fused = self.fusion(aligned, lab_seq)
        return self.seq_head(fused, lab_lengths)


# Backward compatibility: previous spelling of the class name.
TNformerDiseasePredictor = TnformerDiseasePredictor
