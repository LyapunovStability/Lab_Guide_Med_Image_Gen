"""项目根目录与默认本地 Hugging Face 快照路径（checkpoints/ 下）。"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

_PROJECT_ROOT = Path(__file__).resolve().parent.parent


def project_root() -> Path:
    return _PROJECT_ROOT


def checkpoints_dir() -> Path:
    """默认权重根目录：<项目根>/checkpoints"""
    return _PROJECT_ROOT / "checkpoints"


def default_local_sd_snapshot_path() -> Path:
    return checkpoints_dir() / "stable-diffusion"


def default_local_pubmedbert_path() -> Path:
    return checkpoints_dir() / "pubmedbert-base-embeddings"


def resolve_sd_model_source(value: Optional[str]) -> str:
    """
    解析 Stable Diffusion 根目录（含 vae/unet/scheduler 子目录）。

    - None 或空字符串 -> <项目根>/checkpoints/stable-diffusion（绝对路径）
    - 已是存在的绝对路径目录 -> 规范化后的绝对路径
    - 相对项目根的路径且目录存在 -> 绝对路径
    - 否则视为 Hugging Face Hub 模型 ID（如 runwayml/stable-diffusion-v1-5）
    """
    default = default_local_sd_snapshot_path()
    if value is None or not str(value).strip():
        return str(default.resolve())
    v = str(value).strip()
    p = Path(v)
    if p.is_absolute():
        return str(p.resolve()) if p.is_dir() else v
    rel = (_PROJECT_ROOT / v).resolve()
    if rel.is_dir():
        return str(rel)
    return v


def resolve_pubmedbert_source(value: Optional[str]) -> str:
    """
    解析 PubMedBERT（或兼容的 AutoModel）目录。

    规则同 resolve_sd_model_source，默认目录为 checkpoints/pubmedbert-base-embeddings。
    """
    default = default_local_pubmedbert_path()
    if value is None or not str(value).strip():
        return str(default.resolve())
    v = str(value).strip()
    p = Path(v)
    if p.is_absolute():
        return str(p.resolve()) if p.is_dir() else v
    rel = (_PROJECT_ROOT / v).resolve()
    if rel.is_dir():
        return str(rel)
    return v
