# Laboratory Test-Guided Medical Image Generation for Multi-Modal Disease Prediction

This repository contains the implementation of **Laboratory Test-Guided Medical Image Generation for Multi-Modal Disease Prediction**.

Integrating laboratory tests and medical images is crucial for accurate disease prediction. However, medical imaging data is **temporally sparse** compared to frequently collected laboratory tests, which limits effective multi-modal interaction. We mitigate this issue by **generating additional CXR images at more time points**, conditioned on the laboratory test time series.

Inspired by the pivotal role of **organs** in mediating laboratory tests and imaging abnormalities, we propose an **Organ-Centric Modal-Shared Image Generator**: an **Organ-Centric Graph** connects laboratory tests, organs, and imaging abnormalities, while a **Knowledge-Guided Modal-Shared Trajectory Module** binds cross-time multi-modal features into a shared organ-state trajectory to infer abnormality features for **diffusion-based image generation** (with structural priors from the nearest observed image).

## Project Structure (Overview)

> The core logical structure of this project is shown below (subject to your local codebase).

```text
Lab_Guided_Med_Image_Syn
|- checkpoints/                  # Downloaded pretrained weights (image encoder, diffusion, text encoder etc.)
|- configs/                      # Stage 1 / Stage 2 / inference YAML configs
|- data/                         # Example pickles and data layout (see data_example.py)
|- datasets/                     # DataLoader and collate logic
|- graph/                        # Organ graph assets, construction scripts, prior knowledge
|- models/                       # Encoders, trajectory, organ graph modules, diffusion pipeline
|- prediction/                   # Post-generation disease prediction (configs, models, train/infer)
|- utils/                        # Helpers (time-point selection, EVA-X checkpoint, project paths)
|- train.py                      # Image generator training (Stage 1 / Stage 2)
|- inference.py                  # Image generator inference entry point
|- run_simulated_training.py     # Simulated-data training smoke test
|- smoke_test.py                 # Dependency/file checks + optional smoke training
`- requirements.txt              # Python dependencies
```

## Environment Setup

Required versions:

- Python: `3.10`
- PyTorch: `2.4.x` (recommended: `torch==2.4.0`, `torchvision==0.19.0`)

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Pretrained Model Checkpoints

Training and inference expect **third-party pretrained weights**, including (as applicable) the **text encoder**, **image encoder**, and **diffusion** checkpoints.

Download the bundle from Baidu Netdisk:

- [Pretrained checkpoints](https://pan.baidu.com/s/1kRT0zxeAj_oq8qq9V6sVtg?pwd=7c5t) (extraction code: `7c5t`)

After downloading, extract or copy the files into the repository’s **`checkpoints/`** directory so default config paths resolve correctly.


## Data Preparation

### 1. Raw data (MIMIC)

This project builds on **MIMIC-CXR** (chest X-rays) and **MIMIC-IV** (laboratory tests and clinical tables). Both require a PhysioNet account and credentialed access:

- [MIMIC-CXR v2.0.0](https://physionet.org/content/mimic-cxr/2.0.0/)
- [MIMIC-IV](https://physionet.org/content/mimiciv/)

Scripts that turn official MIMIC exports into the project’s pickle format are **still being organized**; when they are added, they will live alongside the dataloader under `datasets/` and this section will be updated.

### 3. Preprocessed data (optional shortcut)

You can skip raw preprocessing and download **preprocessed image + lab** assets from the following link:

- Link: [https://pan.baidu.com/s/1usoGAHKI-P_qM0zpqpi3Zg?pwd=74dn](https://pan.baidu.com/s/1usoGAHKI-P_qM0zpqpi3Zg?pwd=74dn)

After downloading, **extract the archive into the project’s `data/` directory** (so paths line up with configs and examples).

For field names, tensor shapes, and per-patient dictionary structure in the preprocessed data, see **`data/data_example.py`** (reference implementation and comments).

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

After Inference Generation, the image generator writes a pickle with synthesized CXRs and their time points (default path: `data/data_for_gen_infer_with_tar_img.pkl`). 

Take disease prediction for the above data based on the two models available in `prediction/` directory:


```bash
# Train & Evaluate the TDsig model
python prediction/train_disease_prediction.py --config prediction/configs/model_tdsig.yaml
python prediction/infer_disease_prediction.py --checkpoint output/prediction_tdsig/best_model.pt \
  --input_pkl data/data_for_gen_infer_with_tar_img.pkl --output_pkl data/data_for_pred_infer_with_outputs.pkl

# Train & Evaluate the TNformer model
python prediction/train_disease_prediction.py --config prediction/configs/model_tnformer.yaml
python prediction/infer_disease_prediction.py --checkpoint output/prediction_tnformer/best_model.pt \
  --input_pkl data/data_for_gen_infer_with_tar_img.pkl --output_pkl data/data_for_pred_infer_with_outputs.pkl
```


----


## Quick Smoke Test (auto-generated simulated data; quick check that training runs in your environment)

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