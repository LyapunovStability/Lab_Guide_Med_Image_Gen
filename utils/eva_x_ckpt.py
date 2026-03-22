"""
EVA-X-B (ViT-B/16) local checkpoint helpers.

Place weights in the project (default path below), or set ``cxr_encoder_weights_path``
in config (absolute or relative to project root).

Official HuggingFace ``eva_x_base_patch16_merged520k_mim.pt`` is often a DeepSpeed bundle
(~1GB). Convert once to a small plain file::

    python tools/convert_eva_x_checkpoint_to_plain.py

Pretrain resolution: 224x224 (see reference_code/cxr_encoder/EVA-X/eva_x.py).
"""
from pathlib import Path
from typing import Any, Dict, Optional

import torch

EVA_X_B_TIMM_NAME = "eva02_base_patch16_xattn_fusedLN_NaiveSwiGLU_subln_RoPE"
EVA_X_INPUT_SIZE = 224

# Default: plain weights from tools/convert_eva_x_checkpoint_to_plain.py
_DEFAULT_WEIGHTS_REL = Path("checkpoints/eva_x/eva_x_base_patch16_merged520k_mim_weights_only.pt")


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def resolve_eva_x_weights_path(weights_path: Optional[str]) -> Path:
    """Absolute path: explicit `weights_path` or default under project root."""
    if weights_path:
        p = Path(weights_path)
        if not p.is_absolute():
            p = _project_root() / p
    else:
        p = _project_root() / _DEFAULT_WEIGHTS_REL
    return p.resolve()


def is_cxr_encoder_finetune_bundle(ckpt: Any) -> bool:
    """True if ``ckpt`` is a CheXpert finetune output (not raw EVA-X only).

    Supported layouts:

    * **Current (EVA-like)**: ``{"model": <backbone like official EVA>, "abnormality_projection": {...}}``
    * **Legacy**: ``{"cxr_encoder_state_dict": {"vision_backbone.*", "abnormality_projection.*", ...}}``
    """
    if not isinstance(ckpt, dict):
        return False
    if isinstance(ckpt.get("model"), dict) and isinstance(ckpt.get("abnormality_projection"), dict):
        if ckpt["abnormality_projection"]:
            return True
    inner = ckpt.get("cxr_encoder_state_dict")
    if isinstance(inner, dict) and inner:
        return any(k.startswith("vision_backbone.") for k in inner)
    return False


def load_backbone_state_dict(ckpt: Any) -> Dict[str, torch.Tensor]:
    if not isinstance(ckpt, dict):
        raise TypeError("checkpoint must be a dict")
    if "model" in ckpt:
        state = ckpt["model"]
    elif "state_dict" in ckpt:
        state = ckpt["state_dict"]
    else:
        state = ckpt
    if not isinstance(state, dict):
        state = dict(state)
    out: Dict[str, torch.Tensor] = {}
    for k, v in state.items():
        nk = k[7:] if k.startswith("module.") else k
        out[nk] = v
    return out


def apply_pretrained_to_eva_model(
    model: torch.nn.Module, state_dict: Dict[str, torch.Tensor]
) -> None:
    checkpoint_model = dict(state_dict)
    model_sd = model.state_dict()
    for k in list(checkpoint_model.keys()):
        if k in ("head.weight", "head.bias"):
            if k not in model_sd or checkpoint_model[k].shape != model_sd[k].shape:
                checkpoint_model.pop(k, None)
    msg = model.load_state_dict(checkpoint_model, strict=False)
    print("EVA-X-B load_state_dict:", msg)
