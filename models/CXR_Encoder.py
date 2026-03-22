import pytorch_lightning as pl
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

import utils.models_eva  # noqa: F401 — registers EVA timm models
from utils.eva_x_ckpt import (
    EVA_X_B_TIMM_NAME,
    EVA_X_INPUT_SIZE,
    apply_pretrained_to_eva_model,
    is_cxr_encoder_finetune_bundle,
    load_backbone_state_dict,
    resolve_eva_x_weights_path,
)


class CXR_Encoder(pl.LightningModule):
    """Chest X-ray encoder fixed to EVA-X-B (ViT-B/16).

    Model card and pretrained checkpoints: https://huggingface.co/MapleF/eva_x
    (default local file: ``eva_x_base_patch16_merged520k_mim_weights_only.pt``; see ``cxr_encoder_weights_path`` in config).

    You may also set ``cxr_encoder_weights_path`` to the ``.pt`` from
    ``finetune_cxr_encoder_chexpert.py``: same layout as official EVA (key ``model`` for the ViT)
    plus ``abnormality_projection`` for the MLP head. Legacy bundles with
    ``cxr_encoder_state_dict`` are still accepted (``num_abnormalities`` / ``feature_dim`` must match).
    """

    def __init__(
        self,
        task="na",
        pretrained=True,
        num_abnormalities=12,
        feature_dim=512,
        weights_path=None,
    ):
        super().__init__()

        self.task = task
        self.num_abnormalities = num_abnormalities
        self.feature_dim = feature_dim
        self._eva_input_size = EVA_X_INPUT_SIZE

        raw_ckpt = None
        if pretrained:
            ckpt_file = resolve_eva_x_weights_path(weights_path)
            if not ckpt_file.is_file():
                raise FileNotFoundError(
                    f"EVA-X-B weights not found: {ckpt_file}. "
                    "Download checkpoint into the default folder "
                    "or set cxr_encoder_weights_path in config."
                )
            raw_ckpt = torch.load(ckpt_file, map_location="cpu")

        self.vision_backbone = timm.create_model(
            EVA_X_B_TIMM_NAME,
            pretrained=False,
            num_classes=0,
            img_size=EVA_X_INPUT_SIZE,
            use_mean_pooling=True,
            in_chans=3,
        )

        if pretrained and raw_ckpt is not None and not is_cxr_encoder_finetune_bundle(raw_ckpt):
            state_dict = load_backbone_state_dict(raw_ckpt)
            apply_pretrained_to_eva_model(self.vision_backbone, state_dict)

        with torch.no_grad():
            dummy_input = torch.zeros(1, 3, EVA_X_INPUT_SIZE, EVA_X_INPUT_SIZE)
            backbone_feat = self.vision_backbone(dummy_input)
            backbone_dim = (
                backbone_feat.shape[1]
                if len(backbone_feat.shape) > 1
                else backbone_feat.shape[-1]
            )

        # Task-specific head (not in EVA-X ckpt). Trained in Stage 1 while the ViT stays frozen;
        # Stage 2 should load these weights from the Stage 1 checkpoint.
        self.abnormality_projection = nn.Sequential(
            nn.Linear(backbone_dim, feature_dim * num_abnormalities),
            nn.ReLU(),
            nn.Linear(feature_dim * num_abnormalities, num_abnormalities * feature_dim),
        )

        if pretrained and raw_ckpt is not None and is_cxr_encoder_finetune_bundle(raw_ckpt):
            if isinstance(raw_ckpt.get("model"), dict) and isinstance(
                raw_ckpt.get("abnormality_projection"), dict
            ):
                bb_sd = load_backbone_state_dict(raw_ckpt)
                apply_pretrained_to_eva_model(self.vision_backbone, bb_sd)
                msg = self.abnormality_projection.load_state_dict(
                    raw_ckpt["abnormality_projection"], strict=False
                )
                print("CXR_Encoder (model + abnormality_projection) load:", msg)
            else:
                inner = raw_ckpt["cxr_encoder_state_dict"]
                msg = self.load_state_dict(inner, strict=False)
                print("CXR_Encoder (legacy cxr_encoder_state_dict) load_state_dict:", msg)

    def forward(self, x):
        """
        Extract imaging abnormality features from CXR images.

        Args:
            x: Input images (B, 3, H, W). Resized internally to EVA-X pretrain size (224)
               before the ViT; pipeline may still use e.g. 512 for diffusion.

        Returns:
            features: (batch_size, num_abnormalities, feature_dim)
        """
        h, w = x.shape[-2:]
        if h != self._eva_input_size or w != self._eva_input_size:
            x = F.interpolate(
                x,
                size=(self._eva_input_size, self._eva_input_size),
                mode="bilinear",
                align_corners=False,
            )
        backbone_feat = self.vision_backbone(x)

        if len(backbone_feat.shape) == 4:
            backbone_feat = torch.mean(backbone_feat, dim=(2, 3))
        elif len(backbone_feat.shape) == 3:
            backbone_feat = torch.mean(backbone_feat, dim=1)

        flat_features = self.abnormality_projection(backbone_feat)
        features = flat_features.view(
            backbone_feat.shape[0], self.num_abnormalities, self.feature_dim
        )
        return features
