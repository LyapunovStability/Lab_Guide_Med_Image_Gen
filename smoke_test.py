from __future__ import annotations

import argparse
import importlib
import subprocess
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


IMPORT_NAME_OVERRIDES: Dict[str, str] = {
    "pytorch-lightning": "pytorch_lightning",
    "pyyaml": "yaml",
    "pillow": "PIL",
    "opencv-python-headless": "cv2",
    "scikit-learn": "sklearn",
    "huggingface-hub": "huggingface_hub",
}


def unique_keep_order(items: Iterable[str]) -> List[str]:
    seen = set()
    result: List[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result


def parse_requirement_name(line: str) -> str | None:
    raw = line.strip()
    if not raw or raw.startswith("#"):
        return None

    # Drop inline comments / markers / extras / version constraints.
    raw = raw.split("#", 1)[0].strip()
    raw = raw.split(";", 1)[0].strip()
    raw = raw.split("[", 1)[0].strip()
    for sep in ("==", ">=", "<=", "~=", "!=", ">", "<"):
        if sep in raw:
            raw = raw.split(sep, 1)[0].strip()
            break

    return raw or None


def requirement_to_import_name(requirement_name: str) -> str:
    normalized = requirement_name.lower().replace("_", "-")
    if normalized in IMPORT_NAME_OVERRIDES:
        return IMPORT_NAME_OVERRIDES[normalized]
    return requirement_name.replace("-", "_")


def modules_from_requirements(root: Path) -> List[str]:
    requirements_path = root / "requirements.txt"
    if not requirements_path.exists():
        return []

    modules: List[str] = []
    for line in requirements_path.read_text(encoding="utf-8").splitlines():
        req_name = parse_requirement_name(line)
        if not req_name:
            continue
        modules.append(requirement_to_import_name(req_name))

    # torch / torchvision are required by this project, but may not be pinned here.
    modules.extend(["torch", "torchvision"])
    return unique_keep_order(modules)


def check_imports(modules: Iterable[str]) -> Tuple[List[str], List[str]]:
    ok, missing = [], []
    for name in modules:
        try:
            importlib.import_module(name)
            ok.append(name)
        except Exception:
            missing.append(name)
    return ok, missing


def check_paths(root: Path, relative_paths: Iterable[str]) -> Tuple[List[str], List[str]]:
    exists, missing = [], []
    for rel in relative_paths:
        if (root / rel).exists():
            exists.append(rel)
        else:
            missing.append(rel)
    return exists, missing


def run_command(cmd: List[str], cwd: Path) -> int:
    print("\n[Run]", subprocess.list2cmdline(cmd))
    proc = subprocess.run(cmd, cwd=str(cwd))
    return proc.returncode


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Quick smoke test for Med Lab Image Generator."
    )
    parser.add_argument(
        "--project-root",
        type=str,
        default=str(Path(__file__).resolve().parent),
        help="Project root path.",
    )
    parser.add_argument(
        "--run-training-smoke",
        action="store_true",
        help="Run lightweight training smoke test via run_simulated_training.py.",
    )
    parser.add_argument(
        "--stage",
        choices=["1", "2", "all"],
        default="1",
        help="Stage to run when --run-training-smoke is enabled.",
    )
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--limit-train-batches", type=str, default="1")
    parser.add_argument("--limit-val-batches", type=str, default="1")
    parser.add_argument(
        "--use-vision-pretrained",
        action="store_true",
        help="Enable pretrained vision weights in smoke test.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = Path(args.project_root).resolve()

    print("=== Med Lab Image Generator: Smoke Test ===")
    print(f"Project root: {root}")
    print(f"Python executable: {sys.executable}")

    modules_to_check = modules_from_requirements(root)
    if not modules_to_check:
        modules_to_check = [
            "torch",
            "torchvision",
            "pytorch_lightning",
            "diffusers",
            "transformers",
            "yaml",
            "numpy",
        ]
    ok_imports, missing_imports = check_imports(modules_to_check)

    print("\n[Import Check]")
    print("OK     :", ", ".join(ok_imports) if ok_imports else "(none)")
    print("Missing:", ", ".join(missing_imports) if missing_imports else "(none)")

    required_paths = [
        "requirements.txt",
        "smoke_test.py",
        "train.py",
        "inference.py",
        "configs/stage1_config.yaml",
        "configs/stage2_config.yaml",
    ]
    optional_paths = [
        "run_simulated_training.py",
        "configs/inference_config.yaml",
        "graph/organ_graph.json",
        "scripts/select_time_points.py",
        "utils/select_time_points.py",
    ]

    required_exists, required_missing = check_paths(root, required_paths)
    optional_exists, optional_missing = check_paths(root, optional_paths)

    print("\n[File Check]")
    print("Required exists :", ", ".join(required_exists) if required_exists else "(none)")
    print("Required missing:", ", ".join(required_missing) if required_missing else "(none)")
    print("Optional exists :", ", ".join(optional_exists) if optional_exists else "(none)")
    print("Optional missing:", ", ".join(optional_missing) if optional_missing else "(none)")

    if missing_imports:
        print("\n[Result] FAIL: missing Python dependencies.")
        return 1

    if required_missing:
        print("\n[Result] FAIL: missing required project files.")
        return 1

    if args.run_training_smoke:
        runner = root / "run_simulated_training.py"
        if not runner.exists():
            print("\n[Result] FAIL: run_simulated_training.py not found.")
            return 1

        smoke_prerequisites = [
            "data/data_example.py",
            "configs/stage1_config.yaml",
            "configs/stage2_config.yaml",
        ]
        smoke_exists, smoke_missing = check_paths(root, smoke_prerequisites)
        if smoke_missing:
            print("\n[Training Smoke Prerequisite Check]")
            print("Exists :", ", ".join(smoke_exists) if smoke_exists else "(none)")
            print("Missing:", ", ".join(smoke_missing))
            print("\n[Result] FAIL: missing files required by run_simulated_training.py.")
            return 1

        cmd = [
            sys.executable,
            str(runner),
            "--stage",
            args.stage,
            "--epochs",
            str(args.epochs),
            "--batch-size",
            str(args.batch_size),
            "--num-workers",
            str(args.num_workers),
            "--limit-train-batches",
            args.limit_train_batches,
            "--limit-val-batches",
            args.limit_val_batches,
        ]
        if not args.use_vision_pretrained:
            cmd.append("--disable-vision-pretrained")

        rc = run_command(cmd, cwd=root)
        if rc != 0:
            print(f"\n[Result] FAIL: training smoke test exited with code {rc}.")
            return rc

    print("\n[Result] PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
