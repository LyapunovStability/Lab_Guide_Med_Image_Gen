from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np
import torch


def _validate_binary_labels(y_true: np.ndarray) -> np.ndarray:
    y = np.asarray(y_true, dtype=np.float64).reshape(-1)
    return (y > 0.5).astype(np.int64)


def binary_auroc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    y = _validate_binary_labels(y_true)
    scores = np.asarray(y_score, dtype=np.float64).reshape(-1)
    if y.shape[0] != scores.shape[0]:
        raise ValueError("y_true and y_score must have the same length.")

    pos_count = int(y.sum())
    neg_count = int(y.shape[0] - pos_count)
    if pos_count == 0 or neg_count == 0:
        return float("nan")

    order = np.argsort(-scores, kind="mergesort")
    y = y[order]
    scores = scores[order]

    distinct_idxs = np.where(np.diff(scores))[0]
    threshold_idxs = np.r_[distinct_idxs, y.shape[0] - 1]

    tps = np.cumsum(y)[threshold_idxs]
    fps = (1 + threshold_idxs) - tps

    tpr = np.r_[0.0, tps / pos_count]
    fpr = np.r_[0.0, fps / neg_count]
    return float(np.trapz(tpr, fpr))


def binary_auprc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    y = _validate_binary_labels(y_true)
    scores = np.asarray(y_score, dtype=np.float64).reshape(-1)
    if y.shape[0] != scores.shape[0]:
        raise ValueError("y_true and y_score must have the same length.")

    pos_count = int(y.sum())
    if pos_count == 0:
        return float("nan")

    order = np.argsort(-scores, kind="mergesort")
    y = y[order]

    tps = np.cumsum(y)
    fps = np.cumsum(1 - y)
    precision = tps / (tps + fps)
    recall = tps / pos_count

    precision = np.r_[1.0, precision]
    recall = np.r_[0.0, recall]
    return float(np.trapz(precision, recall))


def compute_multilabel_auroc_auprc(probabilities: torch.Tensor, labels: torch.Tensor) -> Dict[str, object]:
    probs = probabilities.detach().cpu().numpy()
    true = labels.detach().cpu().numpy()

    if probs.ndim != 2 or true.ndim != 2:
        raise ValueError("Expected 2D tensors for probabilities and labels.")
    if probs.shape != true.shape:
        raise ValueError("probabilities and labels must have the same shape.")

    per_label_auroc: List[float] = []
    per_label_auprc: List[float] = []
    for label_idx in range(true.shape[1]):
        label_true = true[:, label_idx]
        label_prob = probs[:, label_idx]
        per_label_auroc.append(binary_auroc(label_true, label_prob))
        per_label_auprc.append(binary_auprc(label_true, label_prob))

    macro_auroc = float(np.nanmean(per_label_auroc)) if np.isfinite(per_label_auroc).any() else float("nan")
    macro_auprc = float(np.nanmean(per_label_auprc)) if np.isfinite(per_label_auprc).any() else float("nan")

    return {
        "macro_auroc": macro_auroc,
        "macro_auprc": macro_auprc,
        "per_label_auroc": per_label_auroc,
        "per_label_auprc": per_label_auprc,
    }


def format_metric(value: float) -> str:
    if not np.isfinite(value):
        return "nan"
    return f"{value:.4f}"


def resolve_disease_names(num_diseases: int, configured_names: object = None) -> List[str]:
    if isinstance(configured_names, str):
        parsed = [name.strip() for name in configured_names.split(",") if name.strip()]
        if len(parsed) >= num_diseases:
            return parsed[:num_diseases]
    if isinstance(configured_names, Sequence) and not isinstance(configured_names, (str, bytes)):
        parsed = [str(name) for name in configured_names]
        if len(parsed) >= num_diseases:
            return parsed[:num_diseases]

    if num_diseases == 4:
        return ["Mortality", "Sepsis", "Respiratory Failure", "Heart Failure"]
    return [f"disease_{idx}" for idx in range(num_diseases)]


def format_per_label_metrics(
    per_label_auroc: Sequence[float],
    per_label_auprc: Sequence[float],
    disease_names: Sequence[str],
) -> List[str]:
    label_count = min(len(per_label_auroc), len(per_label_auprc), len(disease_names))
    lines = []
    for idx in range(label_count):
        lines.append(
            f"{disease_names[idx]}: "
            f"AUROC={format_metric(float(per_label_auroc[idx]))}, "
            f"AUPRC={format_metric(float(per_label_auprc[idx]))}"
        )
    return lines
