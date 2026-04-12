import os
import pickle
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class PredictionDataset(Dataset):
    """
    Dataset for multimodal disease prediction (e.g. TDSig, TNformer).

    Required fields per patient sample:
    - lab_test_data, lab_test_time, lab_test_mask
    - reference_image_path (relative paths join reference_image_root when set, else pickle dir / cwd)
    - target_image_path_list (or backward-compatible target_image_path); prefer absolute paths
    - reference_image_time, target_image_time_list (aligned with paths; required)
    - disease_prediction_label (required only when require_label=True)
    """

    def __init__(
        self,
        data_pkl_path: str,
        resize: int = 512,
        crop: int = 512,
        require_label: bool = True,
        reference_image_root: Optional[str] = None,
    ) -> None:
        self.data_pkl_path = data_pkl_path
        self.resize = resize
        self.crop = crop
        self.require_label = require_label
        self.base_dir = os.path.dirname(os.path.abspath(self.data_pkl_path))
        self.reference_image_root = (
            os.path.abspath(reference_image_root)
            if reference_image_root
            else None
        )

        with open(self.data_pkl_path, "rb") as f:
            self.data: Dict[str, Dict[str, Any]] = pickle.load(f)

        self.patient_ids = list(self.data.keys())
        self.image_mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(3, 1, 1)
        self.image_std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(3, 1, 1)

    def __len__(self) -> int:
        return len(self.patient_ids)

    def _resolve_reference_image_path(self, image_path: Optional[str]) -> str:
        """Resolve reference_image_path; optional reference_image_root prepends relative paths."""
        if image_path is None or image_path == "":
            raise FileNotFoundError("Image path is empty.")

        candidates: List[str] = []
        if os.path.isabs(image_path):
            candidates.append(image_path)
        else:
            if self.reference_image_root:
                candidates.append(os.path.join(self.reference_image_root, image_path))
            candidates.append(os.path.join(self.base_dir, image_path))
            candidates.append(image_path)

        for path in candidates:
            if os.path.exists(path):
                return os.path.abspath(path)

        raise FileNotFoundError(
            f"Reference image path '{image_path}' does not exist. Checked: {candidates}"
        )

    def _resolve_target_image_path(self, image_path: Optional[str]) -> str:
        """Resolve generated/target image paths (expected absolute; no reference_image_root)."""
        if image_path is None or image_path == "":
            raise FileNotFoundError("Image path is empty.")

        candidates: List[str] = []
        if os.path.isabs(image_path):
            candidates.append(image_path)
        else:
            candidates.append(os.path.join(self.base_dir, image_path))
            candidates.append(image_path)

        for path in candidates:
            if os.path.exists(path):
                return os.path.abspath(path)

        raise FileNotFoundError(
            f"Target image path '{image_path}' does not exist (use absolute paths or "
            f"paths relative to the pickle directory). Checked: {candidates}"
        )

    def _load_reference_image(self, image_path: str) -> torch.Tensor:
        resolved_path = self._resolve_reference_image_path(image_path)
        image = Image.open(resolved_path).convert("RGB")
        return self._transform_image(image)

    def _load_target_image(self, image_path: str) -> torch.Tensor:
        resolved_path = self._resolve_target_image_path(image_path)
        image = Image.open(resolved_path).convert("RGB")
        return self._transform_image(image)

    def _transform_image(self, image: Image.Image) -> torch.Tensor:
        image = image.resize((self.resize, self.resize), Image.BILINEAR)

        crop_size = min(self.crop, self.resize)
        left = max((image.width - crop_size) // 2, 0)
        top = max((image.height - crop_size) // 2, 0)
        right = left + crop_size
        bottom = top + crop_size
        image = image.crop((left, top, right, bottom))

        # Convert PIL image to tensor without torchvision dependency.
        image_data = torch.tensor(list(image.getdata()), dtype=torch.float32)
        image_tensor = image_data.view(image.height, image.width, 3).permute(2, 0, 1) / 255.0
        image_tensor = (image_tensor - self.image_mean) / self.image_std
        return image_tensor

    def _extract_generated_paths(self, sample: Dict[str, Any]) -> List[str]:
        paths, _ = self._extract_generated_paths_and_times(sample, ref_time_scalar=0.0)
        return paths

    def _extract_generated_paths_and_times(
        self,
        sample: Dict[str, Any],
        ref_time_scalar: float,
    ) -> tuple[List[str], List[float]]:
        """
        Return filtered generated image paths and aligned times (same length).
        Times come from target_image_time_list when present and length-matched;
        otherwise each valid path uses ref_time_scalar as a fallback.
        """
        generated_paths = sample.get("target_image_path_list", None)
        if generated_paths is None:
            single_path = sample.get("target_image_path", None)
            generated_paths = [single_path] if single_path is not None else []

        if isinstance(generated_paths, np.ndarray):
            generated_paths = generated_paths.tolist()
        elif not isinstance(generated_paths, list):
            generated_paths = list(generated_paths)

        times_raw = sample.get("target_image_time_list", None)
        if times_raw is not None:
            if isinstance(times_raw, np.ndarray):
                times_raw = times_raw.tolist()
            elif not isinstance(times_raw, list):
                times_raw = list(times_raw)
        else:
            times_raw = []

        if times_raw and len(times_raw) != len(generated_paths):
            raise ValueError(
                "target_image_time_list length must match target_image_path_list "
                f"({len(times_raw)} vs {len(generated_paths)})."
            )

        single_target_time = sample.get("target_image_time", None)

        filtered_paths: List[str] = []
        filtered_times: List[float] = []
        for idx, path in enumerate(generated_paths):
            if path is None or path == "":
                continue
            filtered_paths.append(path)
            if idx < len(times_raw) and times_raw[idx] is not None:
                filtered_times.append(float(times_raw[idx]))
            elif len(filtered_paths) == 1 and single_target_time is not None:
                filtered_times.append(float(single_target_time))
            else:
                filtered_times.append(float(ref_time_scalar))

        return filtered_paths, filtered_times

    def __getitem__(self, index: int) -> Dict[str, Any]:
        patient_id = self.patient_ids[index]
        sample = self.data[patient_id]

        required_keys = [
            "lab_test_data",
            "lab_test_time",
            "lab_test_mask",
            "reference_image_path",
        ]
        for key in required_keys:
            if key not in sample:
                raise KeyError(f"Missing required key '{key}' for patient '{patient_id}'.")

        lab_test = torch.tensor(sample["lab_test_data"], dtype=torch.float32)
        lab_times = torch.tensor(sample["lab_test_time"], dtype=torch.float32)
        lab_mask = torch.tensor(sample["lab_test_mask"], dtype=torch.float32)

        ref_img = self._load_reference_image(sample["reference_image_path"])

        if "reference_image_time" not in sample:
            raise KeyError(
                f"Missing 'reference_image_time' for patient '{patient_id}'. "
                "Required for temporal prediction models (e.g. TNformer)."
            )
        ref_time = torch.tensor(float(sample["reference_image_time"]), dtype=torch.float32)

        generated_paths, generated_times = self._extract_generated_paths_and_times(
            sample,
            ref_time_scalar=float(ref_time.item()),
        )
        generated_images: List[torch.Tensor] = []
        for path in generated_paths:
            generated_images.append(self._load_target_image(path))

        if generated_images:
            gen_imgs = torch.stack(generated_images, dim=0)
            gen_times = torch.tensor(generated_times, dtype=torch.float32)
        else:
            channels, height, width = ref_img.shape
            gen_imgs = torch.empty((0, channels, height, width), dtype=ref_img.dtype)
            gen_times = torch.empty((0,), dtype=torch.float32)

        label = None
        if "disease_prediction_label" in sample:
            label = torch.tensor(sample["disease_prediction_label"], dtype=torch.float32)
        elif self.require_label:
            raise KeyError(
                f"Missing required key 'disease_prediction_label' for patient '{patient_id}'."
            )

        return {
            "patient_id": patient_id,
            "lab_test": lab_test,
            "lab_times": lab_times,
            "lab_mask": lab_mask,
            "lab_length": lab_test.shape[0],
            "ref_img": ref_img,
            "gen_imgs": gen_imgs,
            "gen_img_count": gen_imgs.shape[0],
            "ref_time": ref_time,
            "gen_times": gen_times,
            "label": label,
        }


class PredictionCollate:
    """
    Collate function that pads variable-length lab sequences and variable-size
    generated image lists.
    """

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        batch_size = len(batch)
        if batch_size == 0:
            raise ValueError("PredictionCollate received an empty batch.")

        # 1) Pad lab sequence fields
        max_lab_len = max(item["lab_length"] for item in batch)
        feat_dim = batch[0]["lab_test"].shape[1]

        batch_lab_test = torch.zeros(batch_size, max_lab_len, feat_dim, dtype=torch.float32)
        batch_lab_mask = torch.zeros(batch_size, max_lab_len, feat_dim, dtype=torch.float32)
        batch_lab_times = torch.zeros(batch_size, max_lab_len, dtype=torch.float32)
        lab_lengths = torch.zeros(batch_size, dtype=torch.long)

        for i, item in enumerate(batch):
            curr_len = item["lab_length"]
            batch_lab_test[i, :curr_len] = item["lab_test"]
            batch_lab_mask[i, :curr_len] = item["lab_mask"]
            batch_lab_times[i, :curr_len] = item["lab_times"]
            lab_lengths[i] = curr_len

        # 2) Stack reference image fields
        batch_ref_img = torch.stack([item["ref_img"] for item in batch], dim=0)
        channels, height, width = batch_ref_img.shape[1:]

        # 3) Pad generated image list fields
        max_gen_count = max(item["gen_img_count"] for item in batch)
        max_gen_count = max(max_gen_count, 1)  # Keep tensor shape stable when no generated image exists.

        batch_gen_imgs = torch.zeros(
            batch_size,
            max_gen_count,
            channels,
            height,
            width,
            dtype=batch_ref_img.dtype,
        )
        batch_gen_img_mask = torch.zeros(batch_size, max_gen_count, dtype=torch.bool)
        batch_gen_times = torch.zeros(batch_size, max_gen_count, dtype=torch.float32)
        gen_img_counts = torch.zeros(batch_size, dtype=torch.long)
        batch_ref_time = torch.stack([item["ref_time"] for item in batch], dim=0)

        for i, item in enumerate(batch):
            curr_count = item["gen_img_count"]
            gen_img_counts[i] = curr_count
            if curr_count > 0:
                batch_gen_imgs[i, :curr_count] = item["gen_imgs"]
                batch_gen_img_mask[i, :curr_count] = True
                batch_gen_times[i, :curr_count] = item["gen_times"]

        output = {
            "patient_id": [item["patient_id"] for item in batch],
            "lab_test": batch_lab_test,
            "lab_mask": batch_lab_mask,
            "lab_times": batch_lab_times,
            "lab_lengths": lab_lengths,
            "ref_img": batch_ref_img,
            "ref_time": batch_ref_time,
            "gen_imgs": batch_gen_imgs,
            "gen_img_mask": batch_gen_img_mask,
            "gen_times": batch_gen_times,
            "gen_img_count": gen_img_counts,
        }

        labels = [item["label"] for item in batch]
        if all(label is None for label in labels):
            output["label"] = None
        elif any(label is None for label in labels):
            raise ValueError("Batch contains mixed labeled/unlabeled samples.")
        else:
            output["label"] = torch.stack(labels, dim=0)

        return output
