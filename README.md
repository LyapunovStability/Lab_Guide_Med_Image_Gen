# Laboratory Test-Guided Medical Image Generation for Multi-Modal Disease Prediction

This repository contains the implementation of **Laboratory Test-Guided Medical Image Generation for Multi-Modal Disease Prediction**.

Integrating laboratory tests and medical images is crucial for accurate disease prediction. However, medical imaging data is **temporally sparse** compared to frequently collected laboratory tests, which limits effective multi-modal interaction. We mitigate this issue by **generating additional CXR images at more time points**, conditioned on the laboratory test time series.

Inspired by the pivotal role of **organs** in mediating laboratory tests and imaging abnormalities, we propose an **Organ-Centric Modal-Shared Image Generator**: an **Organ-Centric Graph** connects laboratory tests, organs, and imaging abnormalities, while a **Knowledge-Guided Modal-Shared Trajectory Module** binds cross-time multi-modal features into a shared organ-state trajectory to infer abnormality features for **diffusion-based image generation** (with structural priors from the nearest observed image).

## Project Structure (Overview)

> The core logical structure of this project is shown below (subject to your local codebase).

```text
Med_Lab_Image_Generator/
|- configs/                      # Stage1/Stage2/Inference YAML configs
|- datasets/                     # DataLoader and collate logic
|- graph/                        # Organ graph and prior knowledge
|- models/                       # Encoders, trajectory, diffusion pipeline
|- utils/                        # Utility scripts (e.g., time-point selection)
|- train.py                      # Training entry point (Stage 1 / Stage 2)
|- inference.py                  # Inference entry point
|- run_simulated_training.py     # Simulated-data training smoke test
|- smoke_test.py                 # Dependency/file checks + optional smoke training
`- requirements.txt              # Python dependencies
```

## Environment Setup

Default environment is Linux.

Required versions:

- Python: `3.10`
- PyTorch: `2.2.x` (recommended: `torch==2.2.2`, `torchvision==0.17.2`)

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```


## Data Preparation

1. Download the data from the following link:

```bash
https://physionet.org/content/mimic-cxr/2.0.0/
https://physionet.org/content/mimiciv/
```

2. Prepare the data for training and inference.

（正在整理中）

## Image Generator Training

Training in this project has two stages:

- **Stage 1**: Pre-train the Trajectory module
- **Stage 2**: Jointly train Diffusion + Trajectory

### Stage 1

```bash
python train.py --config configs/stage1_config.yaml
```

### Stage 2

```bash
python train.py --config configs/stage2_config.yaml
```

## Image Generator Inference

A typical inference workflow has two steps:

1. Prepare/define target generation time points `T_gen` for each patient.
2. Run `inference.py` to generate images at those time points.

### Target Time-Point Selection 

```bash
python utils/select_time_points.py \
  --lab_test_data_path data/data_for_gen_infer.pkl \
  --output_path data/data_for_gen_infer_with_tar_points.pkl \
  --num_gen_points 3 \
  --device cuda \
  --merge_with_data
```

### Run Inference Generation

```bash
python inference.py \
  --config configs/stage2_config.yaml \
  --checkpoint output/stage2/checkpoints/best_model.ckpt \
  --input_data data/data_for_gen_infer_with_tar_points.pkl \
  --output_dir results/inference \
  --output_pkl data/data_for_gen_infer_with_tar_img.pkl \
  --device cuda
```

## Disease Prediction based on the generated images (after inference)

（正在整理中）

----

## Quick Smoke Test（自动生成模拟数据，快速验证训练流程是否可在当前环境中正常运行）

A new `smoke_test.py` is included for:

- Dependency import checks (`torch` / `lightning` / `diffusers` / `transformers`)
- Existence checks for key scripts and configurations
- Optionally triggering a minimal training smoke test (calls `run_simulated_training.py`)

### Checks Only

```bash
python smoke_test.py
```

### Run Lightweight Training Smoke Test (Stage 1)

```bash
python smoke_test.py --run-training-smoke --stage 1
```

### Run Lightweight Smoke Tests for Stage 1 + Stage 2

```bash
python smoke_test.py --run-training-smoke --stage all
```

## 6) Reference Documents

- Architecture description: `docs/architecture.md`
- Training/inference workflow: `docs/workflow.md`
- Pre-training verification checklist: `docs/verification_checklist.md`