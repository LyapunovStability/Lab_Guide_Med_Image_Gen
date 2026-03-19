from __future__ import annotations

import argparse
from pathlib import Path
from typing import Union

import pytorch_lightning as pl
import torch
import yaml
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from data.data_example import main as generate_simulated_data
from train import ControlGenDataModule, ControlGenModel


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
SIM_TRAIN_FILES = [
    DATA_DIR / "train_data_for_gen_develop.pkl",
    DATA_DIR / "val_data_for_gen_develop.pkl",
    DATA_DIR / "test_data_for_gen_develop.pkl",
]


def parse_limit(value: str) -> Union[int, float]:
    parsed = float(value)
    if parsed >= 1 and parsed.is_integer():
        return int(parsed)
    return parsed


def load_config(config_path: Path) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_config(config: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False, allow_unicode=True)


def ensure_simulated_data(force_regenerate: bool = False) -> None:
    missing_files = [path for path in SIM_TRAIN_FILES if not path.exists()]
    if force_regenerate or missing_files:
        print("Generating simulated training data with data/data_example.py ...")
        generate_simulated_data()


def build_stage_config(
    base_config_path: Path,
    output_root: Path,
    batch_size: int,
    num_workers: int,
    epochs: int,
    disable_vision_pretrained: bool,
) -> dict:
    config = load_config(base_config_path)
    stage_name = f"stage{config['stage']}"

    config["train_lab_test_pkl_path"] = str(DATA_DIR / "train_data_for_gen_develop.pkl")
    config["val_lab_test_pkl_path"] = str(DATA_DIR / "val_data_for_gen_develop.pkl")
    config["test_lab_test_pkl_path"] = str(DATA_DIR / "test_data_for_gen_develop.pkl")
    config["batch_size"] = batch_size
    config["num_workers"] = num_workers
    config["epochs"] = epochs
    config["output_dir"] = str(output_root / stage_name / "checkpoints")
    config["log_dir"] = str(output_root / stage_name / "logs")

    # YAML scientific notation (e.g., "1e-4") may be loaded as strings by some parsers.
    # Normalize known LR fields to numeric values for optimizer compatibility.
    for lr_key in ("learning_rate", "adapter_learning_rate", "unet_learning_rate"):
        if lr_key in config and isinstance(config[lr_key], str):
            config[lr_key] = float(config[lr_key])

    if disable_vision_pretrained:
        config["pretrained"] = False

    return config


def run_stage(
    config: dict,
    limit_train_batches: Union[int, float],
    limit_val_batches: Union[int, float],
) -> str:
    pl.seed_everything(config["seed"])

    dm = ControlGenDataModule(config)
    model = ControlGenModel(config)

    output_dir = Path(config["output_dir"])
    log_dir = Path(config["log_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_callback = ModelCheckpoint(
        dirpath=str(output_dir),
        filename="best_model",
        save_top_k=1,
        monitor=f"val_loss_s{config['stage']}",
        mode="min",
    )

    trainer = pl.Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        max_epochs=config["epochs"],
        callbacks=[checkpoint_callback],
        logger=TensorBoardLogger(save_dir=str(log_dir), name=""),
        limit_train_batches=limit_train_batches,
        limit_val_batches=limit_val_batches,
        num_sanity_val_steps=0,
    )

    trainer.fit(model, dm)

    if checkpoint_callback.best_model_path:
        return checkpoint_callback.best_model_path

    fallback_ckpt = output_dir / "best_model.ckpt"
    return str(fallback_ckpt)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Use simulated data to smoke test Stage 1 / Stage 2 training."
    )
    parser.add_argument(
        "--stage",
        choices=["1", "2", "all"],
        default="all",
        help="Which stage to run. 'all' runs Stage 1 first, then Stage 2.",
    )
    parser.add_argument(
        "--stage1-config",
        type=str,
        default="configs/stage1_config.yaml",
        help="Base config path for Stage 1.",
    )
    parser.add_argument(
        "--stage2-config",
        type=str,
        default="configs/stage2_config.yaml",
        help="Base config path for Stage 2.",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default="output/simulated_smoke_test",
        help="Root folder for generated checkpoints and logs.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=1,
        help="Epochs for each smoke-test stage.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size used for the smoke test.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="Dataloader workers used for the smoke test.",
    )
    parser.add_argument(
        "--limit-train-batches",
        type=parse_limit,
        default=2,
        help="Lightning limit_train_batches. Use 2 for two batches, or 0.25 for 25%%.",
    )
    parser.add_argument(
        "--limit-val-batches",
        type=parse_limit,
        default=1,
        help="Lightning limit_val_batches. Use 1 for one batch, or 0.5 for 50%%.",
    )
    parser.add_argument(
        "--force-regenerate-data",
        action="store_true",
        help="Always regenerate simulated data before training.",
    )
    parser.add_argument(
        "--disable-vision-pretrained",
        action="store_true",
        help="Set vision encoder pretrained=false to avoid timm pretrained downloads.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_root = ROOT / args.output_root
    output_root.mkdir(parents=True, exist_ok=True)

    ensure_simulated_data(force_regenerate=args.force_regenerate_data)

    stage1_ckpt_path = None

    if args.stage in {"1", "all"}:
        stage1_config = build_stage_config(
            base_config_path=ROOT / args.stage1_config,
            output_root=output_root,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            epochs=args.epochs,
            disable_vision_pretrained=args.disable_vision_pretrained,
        )
        save_config(stage1_config, output_root / "stage1_effective_config.yaml")

        print("Running Stage 1 smoke test ...")
        stage1_ckpt_path = run_stage(
            config=stage1_config,
            limit_train_batches=args.limit_train_batches,
            limit_val_batches=args.limit_val_batches,
        )
        print(f"Stage 1 checkpoint: {stage1_ckpt_path}")

    if args.stage in {"2", "all"}:
        stage2_config = build_stage_config(
            base_config_path=ROOT / args.stage2_config,
            output_root=output_root,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            epochs=args.epochs,
            disable_vision_pretrained=args.disable_vision_pretrained,
        )

        if stage1_ckpt_path is not None:
            stage2_config["stage1_checkpoint_path"] = stage1_ckpt_path

        save_config(stage2_config, output_root / "stage2_effective_config.yaml")

        print("Running Stage 2 smoke test ...")
        stage2_ckpt_path = run_stage(
            config=stage2_config,
            limit_train_batches=args.limit_train_batches,
            limit_val_batches=args.limit_val_batches,
        )
        print(f"Stage 2 checkpoint: {stage2_ckpt_path}")

    print("Simulated training smoke test finished.")


if __name__ == "__main__":
    main()
