"""Prediction utilities (metrics, model dispatch)."""

from prediction.utils.metrics import (
    binary_auprc,
    binary_auroc,
    compute_multilabel_auroc_auprc,
    format_metric,
    format_per_label_metrics,
    resolve_disease_names,
)
from prediction.utils.model_dispatch import (
    build_prediction_model,
    forward_prediction_model,
    normalize_prediction_model_type,
)

__all__ = [
    "binary_auprc",
    "binary_auroc",
    "build_prediction_model",
    "compute_multilabel_auroc_auprc",
    "format_metric",
    "format_per_label_metrics",
    "forward_prediction_model",
    "normalize_prediction_model_type",
    "resolve_disease_names",
]
