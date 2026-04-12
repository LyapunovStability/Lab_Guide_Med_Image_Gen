import argparse
import os
import pickle
import sys
from pathlib import Path
from typing import Dict, Optional

import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def resolve_project_relative(path: str) -> str:
    """Resolve a path relative to the project root when it is not absolute."""
    p = Path(path)
    if p.is_absolute():
        return str(p)
    return str(PROJECT_ROOT / p)

from prediction.data import PredictionCollate, PredictionDataset
from prediction.utils.metrics import (
    compute_multilabel_auroc_auprc,
    format_per_label_metrics,
    resolve_disease_names,
)
from prediction.utils.model_dispatch import build_prediction_model, forward_prediction_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inference for multimodal disease prediction (TDSig or TNformer).")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to trained prediction checkpoint (.pt).",
    )
    parser.add_argument(
        "--input_pkl",
        type=str,
        default="data/data_for_gen_infer_with_tar_img.pkl",
        help="Input pickle containing generator outputs and original data.",
    )
    parser.add_argument(
        "--output_pkl",
        type=str,
        default="data/data_for_pred_infer_with_outputs.pkl",
        help="Output pickle path with prediction logits/probabilities.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Inference batch size.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=0,
        help="DataLoader workers.",
    )
    parser.add_argument(
        "--resize",
        type=int,
        default=None,
        help="Image resize. If omitted, fallback to checkpoint config or 224.",
    )
    parser.add_argument(
        "--crop",
        type=int,
        default=None,
        help="Image crop. If omitted, fallback to checkpoint config or 224.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device override: cpu/cuda. Default auto-detect.",
    )
    parser.add_argument(
        "--reference_image_root",
        type=str,
        default=None,
        help="Override checkpoint config: directory prepended to relative reference_image_path.",
    )
    return parser.parse_args()


def resolve_device(device_arg: Optional[str]) -> torch.device:
    if device_arg is not None:
        return torch.device(device_arg)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(checkpoint_path: str, device: torch.device) -> Dict:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = build_prediction_model(checkpoint, device)
    return {"model": model, "checkpoint": checkpoint}


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    print(f"Using device: {device}")

    loaded = load_model(args.checkpoint, device)
    model = loaded["model"]
    checkpoint = loaded["checkpoint"]
    num_diseases = int(checkpoint["model_kwargs"].get("num_diseases", 4))
    disease_names = checkpoint.get("disease_names")
    if disease_names is None:
        disease_names = resolve_disease_names(
            num_diseases=num_diseases,
            configured_names=checkpoint.get("config", {}).get("disease_names"),
        )

    ckpt_config = checkpoint.get("config", {})
    resize = args.resize if args.resize is not None else int(ckpt_config.get("resize", 224))
    crop = args.crop if args.crop is not None else int(ckpt_config.get("crop", 224))

    ref_root_raw = (
        args.reference_image_root
        if args.reference_image_root is not None
        else ckpt_config.get("reference_image_root")
    )
    reference_image_root = (
        resolve_project_relative(str(ref_root_raw)) if ref_root_raw else None
    )

    dataset = PredictionDataset(
        data_pkl_path=args.input_pkl,
        resize=resize,
        crop=crop,
        require_label=False,
        reference_image_root=reference_image_root,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        collate_fn=PredictionCollate(),
    )

    with open(args.input_pkl, "rb") as f:
        output_data = pickle.load(f)

    all_probs = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            patient_ids = batch["patient_id"]
            lab_test = batch["lab_test"].to(device)
            lab_mask = batch["lab_mask"].to(device)
            lab_lengths = batch["lab_lengths"].to(device)
            ref_img = batch["ref_img"].to(device)
            gen_imgs = batch["gen_imgs"].to(device)
            gen_img_mask = batch["gen_img_mask"].to(device)

            logits = forward_prediction_model(
                model,
                {
                    "lab_test": lab_test,
                    "lab_mask": lab_mask,
                    "lab_lengths": lab_lengths,
                    "lab_times": batch["lab_times"].to(device),
                    "ref_img": ref_img,
                    "gen_imgs": gen_imgs,
                    "gen_img_mask": gen_img_mask,
                    "ref_time": batch["ref_time"].to(device),
                    "gen_times": batch["gen_times"].to(device),
                },
            )
            probs = torch.sigmoid(logits)
            preds = (probs >= 0.5).float()

            labels = batch["label"]
            if labels is not None:
                labels = labels.to(device)
                all_probs.append(probs.detach().cpu())
                all_labels.append(labels.detach().cpu())

            logits_cpu = logits.cpu().numpy()
            probs_cpu = probs.cpu().numpy()
            preds_cpu = preds.cpu().numpy()

            for idx, patient_id in enumerate(patient_ids):
                if patient_id not in output_data:
                    continue
                output_data[patient_id]["disease_prediction_logits"] = logits_cpu[idx].tolist()
                output_data[patient_id]["disease_prediction_probs"] = probs_cpu[idx].tolist()
                output_data[patient_id]["disease_prediction_pred"] = preds_cpu[idx].tolist()

    output_dir = os.path.dirname(os.path.abspath(args.output_pkl))
    os.makedirs(output_dir, exist_ok=True)
    with open(args.output_pkl, "wb") as f:
        pickle.dump(output_data, f)

    if all_labels:
        metrics = compute_multilabel_auroc_auprc(
            probabilities=torch.cat(all_probs, dim=0),
            labels=torch.cat(all_labels, dim=0),
        )
        print("Inference finished. Per-disease metrics:")
        for line in format_per_label_metrics(
            per_label_auroc=metrics["per_label_auroc"],
            per_label_auprc=metrics["per_label_auprc"],
            disease_names=disease_names,
        ):
            print(f"  {line}")
    else:
        print("Inference finished. No labels found in input for metric calculation.")
    print(f"Saved prediction outputs to: {args.output_pkl}")


if __name__ == "__main__":
    main()
