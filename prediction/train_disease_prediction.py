import argparse
import os
import random
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import numpy as np
import torch
from torch.utils.data import DataLoader, random_split

try:
    import yaml
except ModuleNotFoundError:
    yaml = None
try:
    from tqdm import tqdm
except ModuleNotFoundError:
    tqdm = None
from torch.utils.tensorboard import SummaryWriter

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from prediction.data import PredictionCollate, PredictionDataset
from prediction.models.tdsig import TDSigDiseasePredictor
from prediction.models.tnformer import TnformerDiseasePredictor
from prediction.utils.metrics import (
    compute_multilabel_auroc_auprc,
    format_per_label_metrics,
    resolve_disease_names,
)
from prediction.utils.model_dispatch import (
    forward_prediction_model,
    normalize_prediction_model_type,
)


def resolve_project_relative(path: str) -> str:
    """Resolve a path relative to the project root when it is not absolute."""
    p = Path(path)
    if p.is_absolute():
        return str(p)
    return str(PROJECT_ROOT / p)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_config(config_path: str) -> Dict:
    with open(config_path, "r", encoding="utf-8") as f:
        raw_text = f.read()

    if yaml is not None:
        return yaml.safe_load(raw_text)

    # Fallback parser for flat YAML "key: value" config files.
    def parse_scalar(value: str):
        value = value.strip()
        if value == "":
            return None
        if value.startswith("'") and value.endswith("'"):
            return value[1:-1]
        if value.startswith('"') and value.endswith('"'):
            return value[1:-1]
        lower_value = value.lower()
        if lower_value == "true":
            return True
        if lower_value == "false":
            return False
        try:
            return int(value)
        except ValueError:
            pass
        try:
            return float(value)
        except ValueError:
            pass
        return value

    config: Dict = {}
    for line in raw_text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if ":" not in stripped:
            continue
        key, value = stripped.split(":", 1)
        key = key.strip()
        # Remove inline comments in simple "value # comment" cases.
        value = value.split(" #", 1)[0].strip()
        config[key] = parse_scalar(value)
    return config


def build_dataloader(
    data_path: str,
    batch_size: int,
    num_workers: int,
    resize: int,
    crop: int,
    shuffle: bool,
    reference_image_root: Optional[str] = None,
) -> DataLoader:
    dataset = PredictionDataset(
        data_pkl_path=data_path,
        resize=resize,
        crop=crop,
        require_label=True,
        reference_image_root=reference_image_root,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        collate_fn=PredictionCollate(),
    )


def build_split_dataloaders(
    data_path: str,
    batch_size: int,
    num_workers: int,
    resize: int,
    crop: int,
    seed: int,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    reference_image_root: Optional[str] = None,
) -> Tuple[DataLoader, DataLoader]:
    dataset = PredictionDataset(
        data_pkl_path=data_path,
        resize=resize,
        crop=crop,
        require_label=True,
        reference_image_root=reference_image_root,
    )
    total_size = len(dataset)
    if total_size < 3:
        raise ValueError("At least 3 samples are required for train/val/test split.")

    if train_ratio <= 0 or val_ratio <= 0 or test_ratio < 0:
        raise ValueError("Invalid split ratios. Require train>0, val>0, test>=0.")

    ratio_sum = train_ratio + val_ratio + test_ratio
    if ratio_sum <= 0:
        raise ValueError("Split ratios sum must be positive.")

    train_ratio = train_ratio / ratio_sum
    val_ratio = val_ratio / ratio_sum
    test_ratio = test_ratio / ratio_sum

    train_size = max(1, int(total_size * train_ratio))
    val_size = max(1, int(total_size * val_ratio))
    test_size = total_size - train_size - val_size

    if test_size < 1:
        test_size = 1
        if train_size > val_size and train_size > 1:
            train_size -= 1
        elif val_size > 1:
            val_size -= 1
        else:
            raise ValueError("Dataset is too small to keep non-empty train/val/test splits.")

    split_generator = torch.Generator().manual_seed(seed)
    train_set, val_set, _ = random_split(
        dataset,
        [train_size, val_size, test_size],
        generator=split_generator,
    )

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
        collate_fn=PredictionCollate(),
    )
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        collate_fn=PredictionCollate(),
    )
    return train_loader, val_loader


def move_batch_to_device(batch: Dict, device: torch.device) -> Dict:
    moved = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            moved[key] = value.to(device)
        else:
            moved[key] = value
    return moved


def run_epoch(
    model: Union[TDSigDiseasePredictor, TnformerDiseasePredictor],
    dataloader: DataLoader,
    criterion: torch.nn.Module,
    device: torch.device,
    optimizer: Optional[torch.optim.Optimizer] = None,
    epoch_idx: Optional[int] = None,
    split_name: str = "train",
    enable_tqdm: bool = True,
) -> Tuple[float, Dict[str, object]]:
    is_train = optimizer is not None
    model.train(is_train)

    total_loss = 0.0
    total_batches = 0
    all_probs = []
    all_labels = []

    iterator = dataloader
    if enable_tqdm and tqdm is not None:
        desc = split_name
        if epoch_idx is not None:
            desc = f"Epoch {epoch_idx:03d} [{split_name}]"
        iterator = tqdm(dataloader, desc=desc, dynamic_ncols=True, leave=False)

    for batch_idx, batch in enumerate(iterator, start=1):
        batch = move_batch_to_device(batch, device)
        labels = batch["label"]
        if labels is None:
            raise ValueError("Training/validation batch is missing 'label'.")

        if is_train:
            optimizer.zero_grad()

        logits = forward_prediction_model(model, batch)
        loss = criterion(logits, labels)

        if is_train:
            loss.backward()
            optimizer.step()

        probs = torch.sigmoid(logits.detach())
        all_probs.append(probs.cpu())
        all_labels.append(labels.detach().cpu())
        total_loss += loss.item()
        total_batches += 1

        if enable_tqdm and tqdm is not None and hasattr(iterator, "set_postfix"):
            if batch_idx == 1 or batch_idx % 10 == 0:
                iterator.set_postfix(
                    loss=f"{loss.item():.4f}",
                    avg=f"{(total_loss / total_batches):.4f}",
                )

    if total_batches == 0:
        raise ValueError("Dataloader returned zero batches.")

    metrics = compute_multilabel_auroc_auprc(
        probabilities=torch.cat(all_probs, dim=0),
        labels=torch.cat(all_labels, dim=0),
    )
    return total_loss / total_batches, metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train multimodal disease prediction (TDSig or TNformer).")
    parser.add_argument(
        "--config",
        type=str,
        default="prediction/configs/model_tdsig.yaml",
        help="Path to prediction training config.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    set_seed(int(config.get("seed", 42)))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    output_dir = config.get("output_dir", "output/prediction_tdsig")
    os.makedirs(output_dir, exist_ok=True)

    if config.get("tensorboard_dir"):
        tensorboard_dir = resolve_project_relative(str(config["tensorboard_dir"]))
    elif config.get("log_dir"):
        tensorboard_dir = resolve_project_relative(
            os.path.join(str(config["log_dir"]), "tensorboard")
        )
    else:
        out_leaf = Path(output_dir).name
        tensorboard_dir = str(PROJECT_ROOT / "logs" / "prediction" / out_leaf)
    os.makedirs(tensorboard_dir, exist_ok=True)

    batch_size = int(config.get("batch_size", 4))
    num_workers = int(config.get("num_workers", 0))
    resize = int(config.get("resize", 224))
    crop = int(config.get("crop", 224))
    seed = int(config.get("seed", 42))
    enable_tqdm = bool(config.get("enable_tqdm", True))
    if enable_tqdm and tqdm is None:
        print("tqdm is not installed; falling back to plain loop logging.")
    tb_writer = SummaryWriter(log_dir=tensorboard_dir)
    print(f"TensorBoard logging enabled: {tensorboard_dir}")

    ref_root_raw = config.get("reference_image_root")
    reference_image_root = (
        resolve_project_relative(str(ref_root_raw)) if ref_root_raw else None
    )

    # Preferred mode: split disease-prediction input file in prediction pipeline.
    data_path = config.get("data_path", None)
    if data_path:
        train_loader, val_loader = build_split_dataloaders(
            data_path=data_path,
            batch_size=batch_size,
            num_workers=num_workers,
            resize=resize,
            crop=crop,
            seed=seed,
            train_ratio=float(config.get("train_ratio", 0.7)),
            val_ratio=float(config.get("val_ratio", 0.15)),
            test_ratio=float(config.get("test_ratio", 0.15)),
            reference_image_root=reference_image_root,
        )
        print(f"Using single prediction data file with split: {data_path}")
    else:
        # Backward compatibility: explicit train/val paths.
        train_loader = build_dataloader(
            data_path=config["train_data_path"],
            batch_size=batch_size,
            num_workers=num_workers,
            resize=resize,
            crop=crop,
            shuffle=True,
            reference_image_root=reference_image_root,
        )
        val_loader = None
        if config.get("val_data_path", None):
            val_loader = build_dataloader(
                data_path=config["val_data_path"],
                batch_size=batch_size,
                num_workers=num_workers,
                resize=resize,
                crop=crop,
                shuffle=False,
                reference_image_root=reference_image_root,
            )

    model_type = normalize_prediction_model_type(str(config.get("model_type", "tdsig")))
    lab_d_model = int(config.get("lab_d_model", config.get("hidden_dim", 128)))

    if model_type == "tdsig":
        model_kwargs = {
            "lab_input_dim": int(config.get("lab_input_dim", 53)),
            "hidden_dim": int(config.get("hidden_dim", 128)),
            "lab_d_model": lab_d_model,
            "lab_nhead": int(config.get("lab_nhead", 4)),
            "lab_num_layers": int(config.get("lab_num_layers", 2)),
            "lab_ff_dim": int(config.get("lab_ff_dim", lab_d_model * 2)),
            "lab_pooling": str(config.get("lab_pooling", "mean")),
            "lab_positional_max_len": int(config.get("lab_positional_max_len", 512)),
            "image_feat_dim": int(config.get("image_feat_dim", 128)),
            "fusion_dim": int(config.get("fusion_dim", 128)),
            "num_diseases": int(config.get("num_diseases", 4)),
            "dropout": float(config.get("dropout", 0.1)),
        }
        model = TDSigDiseasePredictor(**model_kwargs).to(device)
    elif model_type == "tnformer":
        tn_ff = config.get("tn_fusion_ff_dim", None)
        model_kwargs = {
            "lab_input_dim": int(config.get("lab_input_dim", 53)),
            "hidden_dim": int(config.get("hidden_dim", 128)),
            "lab_d_model": lab_d_model,
            "lab_nhead": int(config.get("lab_nhead", 4)),
            "lab_num_layers": int(config.get("lab_num_layers", 2)),
            "lab_ff_dim": int(config.get("lab_ff_dim", lab_d_model * 2)),
            "lab_positional_max_len": int(config.get("lab_positional_max_len", 512)),
            "image_feat_dim": int(config.get("image_feat_dim", 128)),
            "fusion_dim": int(config.get("fusion_dim", 128)),
            "num_diseases": int(config.get("num_diseases", 4)),
            "dropout": float(config.get("dropout", 0.1)),
            "tn_num_scales": int(config.get("tn_num_scales", 4)),
            "tn_fusion_nhead": int(config.get("tn_fusion_nhead", 4)),
            "tn_fusion_layers": int(config.get("tn_fusion_layers", 2)),
            "tn_fusion_ff_dim": int(tn_ff) if tn_ff is not None else None,
        }
        model = TnformerDiseasePredictor(**model_kwargs).to(device)
    else:
        raise RuntimeError(f"Unexpected model_type after normalize: {model_type!r}")
    disease_names = resolve_disease_names(
        num_diseases=int(model_kwargs["num_diseases"]),
        configured_names=config.get("disease_names"),
    )

    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config.get("learning_rate", 1e-3)),
        weight_decay=float(config.get("weight_decay", 1e-4)),
    )

    best_val_auprc = float("-inf")
    best_ckpt_path = os.path.join(output_dir, "best_model.pt")
    last_ckpt_path = os.path.join(output_dir, "last_model.pt")

    epochs = int(config.get("epochs", 5))
    for epoch in range(1, epochs + 1):
        train_loss, train_metrics = run_epoch(
            model=model,
            dataloader=train_loader,
            criterion=criterion,
            device=device,
            optimizer=optimizer,
            epoch_idx=epoch,
            split_name="train",
            enable_tqdm=enable_tqdm,
        )

        if val_loader is not None:
            with torch.no_grad():
                val_loss, val_metrics = run_epoch(
                    model=model,
                    dataloader=val_loader,
                    criterion=criterion,
                    device=device,
                    optimizer=None,
                    epoch_idx=epoch,
                    split_name="val",
                    enable_tqdm=enable_tqdm,
                )
        else:
            val_loss, val_metrics = train_loss, train_metrics

        ckpt = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "model_type": model_type,
            "model_kwargs": model_kwargs,
            "disease_names": disease_names,
            "config": config,
            "train_loss": train_loss,
            "train_macro_auroc": train_metrics["macro_auroc"],
            "train_macro_auprc": train_metrics["macro_auprc"],
            "train_per_label_auroc": train_metrics["per_label_auroc"],
            "train_per_label_auprc": train_metrics["per_label_auprc"],
            "val_loss": val_loss,
            "val_macro_auroc": val_metrics["macro_auroc"],
            "val_macro_auprc": val_metrics["macro_auprc"],
            "val_per_label_auroc": val_metrics["per_label_auroc"],
            "val_per_label_auprc": val_metrics["per_label_auprc"],
        }
        torch.save(ckpt, last_ckpt_path)

        val_auprc = val_metrics["macro_auprc"]
        if epoch == 1:
            if np.isfinite(val_auprc):
                best_val_auprc = val_auprc
            torch.save(ckpt, best_ckpt_path)
        elif np.isfinite(val_auprc) and val_auprc > best_val_auprc:
            best_val_auprc = val_auprc
            torch.save(ckpt, best_ckpt_path)

        print(f"Epoch {epoch:03d}")
        for line in format_per_label_metrics(
            per_label_auroc=train_metrics["per_label_auroc"],
            per_label_auprc=train_metrics["per_label_auprc"],
            disease_names=disease_names,
        ):
            print(f"  [train] {line}")
        for line in format_per_label_metrics(
            per_label_auroc=val_metrics["per_label_auroc"],
            per_label_auprc=val_metrics["per_label_auprc"],
            disease_names=disease_names,
        ):
            print(f"  [val]   {line}")

        if tb_writer is not None:
            tb_writer.add_scalar("train/macro_auroc", float(train_metrics["macro_auroc"]), epoch)
            tb_writer.add_scalar("train/macro_auprc", float(train_metrics["macro_auprc"]), epoch)
            tb_writer.add_scalar("val/macro_auroc", float(val_metrics["macro_auroc"]), epoch)
            tb_writer.add_scalar("val/macro_auprc", float(val_metrics["macro_auprc"]), epoch)

            for idx, score in enumerate(train_metrics["per_label_auroc"]):
                tb_writer.add_scalar(f"train/per_label_auroc/{disease_names[idx]}", float(score), epoch)
            for idx, score in enumerate(train_metrics["per_label_auprc"]):
                tb_writer.add_scalar(f"train/per_label_auprc/{disease_names[idx]}", float(score), epoch)
            for idx, score in enumerate(val_metrics["per_label_auroc"]):
                tb_writer.add_scalar(f"val/per_label_auroc/{disease_names[idx]}", float(score), epoch)
            for idx, score in enumerate(val_metrics["per_label_auprc"]):
                tb_writer.add_scalar(f"val/per_label_auprc/{disease_names[idx]}", float(score), epoch)

    if tb_writer is not None:
        tb_writer.close()

    print(f"Training finished. Best checkpoint: {best_ckpt_path}")
    print(f"Last checkpoint: {last_ckpt_path}")


if __name__ == "__main__":
    main()
