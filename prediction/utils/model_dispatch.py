"""Forward pass and checkpoint loading for TDSigDiseasePredictor vs TnformerDiseasePredictor."""

from __future__ import annotations

from typing import Any, Dict

import torch
import torch.nn as nn

from prediction.models.tdsig import TDSigDiseasePredictor
from prediction.models.tnformer import TnformerDiseasePredictor


def forward_prediction_model(model: nn.Module, batch: Dict[str, Any]) -> torch.Tensor:
    if isinstance(model, TnformerDiseasePredictor):
        return model(
            lab_test=batch["lab_test"],
            lab_mask=batch["lab_mask"],
            lab_lengths=batch["lab_lengths"],
            lab_times=batch["lab_times"],
            ref_img=batch["ref_img"],
            gen_imgs=batch["gen_imgs"],
            gen_img_mask=batch["gen_img_mask"],
            ref_time=batch["ref_time"],
            gen_times=batch["gen_times"],
        )
    return model(
        lab_test=batch["lab_test"],
        lab_mask=batch["lab_mask"],
        lab_lengths=batch["lab_lengths"],
        ref_img=batch["ref_img"],
        gen_imgs=batch["gen_imgs"],
        gen_img_mask=batch["gen_img_mask"],
    )


def normalize_prediction_model_type(model_type: str) -> str:
    """Return canonical model_type: ``tdsig`` or ``tnformer``."""
    t = str(model_type).lower().strip()
    if t == "tdsig":
        return "tdsig"
    if t == "tnformer":
        return "tnformer"
    raise ValueError(
        f"model_type must be 'tdsig' or 'tnformer', got {model_type!r}."
    )


def build_prediction_model(
    checkpoint: Dict[str, Any],
    device: torch.device,
) -> nn.Module:
    if "model_kwargs" not in checkpoint:
        raise KeyError("Checkpoint is missing 'model_kwargs'.")
    if "model_state_dict" not in checkpoint:
        raise KeyError("Checkpoint is missing 'model_state_dict'.")

    raw_type = checkpoint.get("model_type", "tdsig")
    model_type = normalize_prediction_model_type(str(raw_type))
    kwargs = checkpoint["model_kwargs"]
    if model_type == "tnformer":
        model = TnformerDiseasePredictor(**kwargs)
    else:
        model = TDSigDiseasePredictor(**kwargs)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model
