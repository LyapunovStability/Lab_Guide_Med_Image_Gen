"""
Offline TimePointSelector: train (Poisson NLL) and/or select generation times T_gen.

**Shell (no subcommands):** provide ``--lab_test_data_path`` and at least one of
``--output_state_path`` (train) or ``--output_path`` (select). If both are set,
the script trains first, then runs selection using the newly saved weights.

  python utils/select_time_points.py \\
      --lab_test_data_path DATA.pkl \\
      --output_state_path selector.pt --output_path OUT.pkl --merge_with_data

Python API: ``run_train_then_select``, ``train_time_point_selector_on_pickle``,
``select_time_points_for_patients``, ``run_time_point_selection`` (yaml-driven).
"""

import argparse
import os
import pickle
import random
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import torch
import yaml
from tqdm import tqdm

from models.TimePointSelector import TimePointSelector


def _lab_presence_mask_1d(lab_mask: Any, device: torch.device) -> torch.Tensor:
    """(L, D) lab mask -> (1, L) float {0,1} indicating any observed lab at that time."""
    if isinstance(lab_mask, np.ndarray):
        lab_mask = torch.from_numpy(lab_mask).float()
    elif not torch.is_tensor(lab_mask):
        lab_mask = torch.as_tensor(lab_mask, dtype=torch.float32)
    lab_mask = lab_mask.to(device)
    if lab_mask.dim() != 2:
        raise ValueError(f"lab_test_mask expected 2D (L, D), got shape {tuple(lab_mask.shape)}")
    present = (lab_mask > 0.5).any(dim=-1).float()
    return present.unsqueeze(0)


def _density_for_times(selector: TimePointSelector, time_points: torch.Tensor) -> torch.Tensor:
    """Same normalization as select_fixed_number / compute_loss."""
    time_min = time_points.min(dim=1, keepdim=True)[0]
    time_max = time_points.max(dim=1, keepdim=True)[0]
    time_range = (time_max - time_min).clamp(min=1e-6)
    time_normalized = (time_points - time_min) / time_range
    return selector.ode_network(time_normalized)


def train_time_point_selector_on_pickle(
    lab_test_data_path: str,
    output_state_path: str,
    epochs: int = 20,
    learning_rate: float = 1e-3,
    device: str = "cuda",
    hidden_dim: int = 64,
    num_ode_layers: int = 3,
    seed: int = 2025,
    max_patients: Optional[int] = None,
    log_prefix: str = "[train_time_selector]",
    print_each_step: bool = False,
) -> None:
    """
    Fit TimePointSelector on lab observation times using Poisson-process NLL in compute_loss.

    Saves a small checkpoint dict loadable by select_time_points_for_patients(selector_state_path=...).
    Trains one patient per step (variable-length sequences); no padding artifacts.
    """
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        device = "cpu"
    dev = torch.device(device)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    with open(lab_test_data_path, "rb") as f:
        lab_test_data: Dict[str, Any] = pickle.load(f)
    patient_ids: List[str] = list(lab_test_data.keys())
    if max_patients is not None:
        patient_ids = patient_ids[: int(max_patients)]

    selector = TimePointSelector(
        hidden_dim=hidden_dim,
        num_ode_layers=num_ode_layers,
    ).to(dev)
    selector.train()
    optimizer = torch.optim.AdamW(selector.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        random.shuffle(patient_ids)
        epoch_losses: List[float] = []
        for patient_id in patient_ids:
            sample = lab_test_data[patient_id]
            time_points = sample.get("lab_test_time")
            lab_mask = sample.get("lab_test_mask")
            if time_points is None or lab_mask is None:
                continue
            if isinstance(time_points, np.ndarray):
                time_points = torch.from_numpy(time_points).float()
            elif not torch.is_tensor(time_points):
                time_points = torch.as_tensor(time_points, dtype=torch.float32)
            if time_points.dim() == 1:
                time_points = time_points.unsqueeze(0)
            time_points = time_points.to(dev)
            if time_points.shape[1] < 1:
                continue

            presence = _lab_presence_mask_1d(lab_mask, dev)
            if presence.sum() < 1e-6:
                continue

            optimizer.zero_grad(set_to_none=True)
            density = _density_for_times(selector, time_points)
            loss = selector.compute_loss(time_points, presence, density)
            if not torch.isfinite(loss):
                continue
            loss.backward()
            optimizer.step()
            lv = float(loss.detach().cpu())
            epoch_losses.append(lv)
            if print_each_step:
                print(
                    f"{log_prefix} epoch {epoch + 1}/{epochs} step patient={patient_id} loss={lv:.6f}",
                    flush=True,
                )

        mean_loss = float(np.mean(epoch_losses)) if epoch_losses else float("nan")
        print(
            f"{log_prefix} epoch {epoch + 1}/{epochs} mean_loss={mean_loss:.6f} n_steps={len(epoch_losses)}",
            flush=True,
        )

    selector.eval()
    out_dir = os.path.dirname(output_state_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    payload = {
        "state_dict": selector.state_dict(),
        "hidden_dim": hidden_dim,
        "num_ode_layers": num_ode_layers,
        "epochs": epochs,
        "learning_rate": learning_rate,
        "seed": seed,
        "lab_test_data_path": lab_test_data_path,
    }
    torch.save(payload, output_state_path)
    print(f"[train_time_selector] saved weights to {output_state_path}")


def select_time_points_for_patients(
    lab_test_data_path: str,
    output_path: str,
    num_gen_points: int = 3,
    hidden_dim: int = 64,
    num_ode_layers: int = 3,
    peak_prominence: float = 0.1,
    peak_distance: int = 1,
    device: str = 'cpu',
    output_format: str = 'pkl',
    merge_with_data: bool = False,
    selector_state_path: Optional[str] = None,
):
    """
    Select generation time points for all patients.
    
    Args:
        lab_test_data_path: Path to lab test pickle file containing patient data
        output_path: Path to save selected time points
        num_gen_points: Number of generation time points (constant across patients)
        hidden_dim: Hidden dimension for ODE network
        num_ode_layers: Number of layers in ODE network
        peak_prominence: Minimum prominence for peak detection
        peak_distance: Minimum distance between peaks
        device: Device to run computation on ('cpu' or 'cuda')
        output_format: Output format ('pkl' or 'csv')
        merge_with_data: If True, merges selected time points into the original data dictionary and saves the full dataset.
        selector_state_path: If set, load TimePointSelector weights from train_time_point_selector_on_pickle checkpoint.
    """
    # Load lab test data
    print(f"Loading lab test data from {lab_test_data_path}...")
    with open(lab_test_data_path, 'rb') as f:
        lab_test_data = pickle.load(f)
    
    patient_ids = list(lab_test_data.keys())
    print(f"Found {len(patient_ids)} patients")
    
    # Initialize time point selector
    time_point_selector = TimePointSelector(
        hidden_dim=hidden_dim,
        num_ode_layers=num_ode_layers,
        peak_prominence=peak_prominence,
        peak_distance=peak_distance
    ).to(device)
    if selector_state_path:
        if not os.path.isfile(selector_state_path):
            raise FileNotFoundError(f"selector_state_path not found: {selector_state_path}")
        ckpt = torch.load(selector_state_path, map_location=device)
        state_dict = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
        ret = time_point_selector.load_state_dict(state_dict, strict=False)
        print(
            f"Loaded TimePointSelector weights from {selector_state_path} "
            f"(missing={len(ret.missing_keys)}, unexpected={len(ret.unexpected_keys)})"
        )
    time_point_selector.eval()
    
    # Dictionary to store T_gen for each patient
    t_gen_dict = {}
    
    print("Selecting generation time points for each patient...")
    with torch.no_grad():
        for patient_id in tqdm(patient_ids):
            patient_data = lab_test_data[patient_id]
            
            # Extract lab data and time points from current project schema.
            lab_data = patient_data.get('lab_test_data')
            time_points = patient_data.get('lab_test_time')
            _mask = patient_data.get('lab_test_mask', None)  # Optional, reserved for future use.

            if lab_data is None or time_points is None:
                available_keys = sorted(list(patient_data.keys()))
                raise KeyError(
                    f"Missing required lab/time fields for patient `{patient_id}`. "
                    "Expected fields: (`lab_test_data`, `lab_test_time`). "
                    f"Available keys: {available_keys}"
                )

            # Convert to tensors
            if isinstance(lab_data, np.ndarray):
                lab_data = torch.from_numpy(lab_data).float()
            elif not torch.is_tensor(lab_data):
                lab_data = torch.as_tensor(lab_data, dtype=torch.float32)

            if isinstance(time_points, np.ndarray):
                time_points = torch.from_numpy(time_points).float()
            elif not torch.is_tensor(time_points):
                time_points = torch.as_tensor(time_points, dtype=torch.float32)
            
            # Ensure time_points is 2D (batch_size=1, seq_length)
            if time_points.dim() == 1:
                time_points = time_points.unsqueeze(0)
            
            # Move to device
            time_points = time_points.to(device)
            
            # Compute density and select time points with fixed size
            density, gen_time_indices = time_point_selector.select_fixed_number(
                time_points=time_points,
                num_gen_points=num_gen_points
            )
            
            # Extract indices for this patient (remove batch dimension)
            gen_indices = gen_time_indices[0].cpu().numpy()  # (num_gen_points,)
            
            # Remove padding values (-1) if any
            gen_indices = gen_indices[gen_indices >= 0]
            
            # Ensure we have exactly num_gen_points
            if len(gen_indices) < num_gen_points:
                # If we have fewer peaks, select top-K by density
                density_1d = density[0].cpu().numpy()
                _, top_indices = torch.topk(
                    torch.from_numpy(density_1d),
                    k=min(num_gen_points, len(density_1d))
                )
                gen_indices = top_indices.numpy()
            
            # Sort indices for consistency
            gen_indices = np.sort(gen_indices[:num_gen_points])
            
            # Store in dictionary
            t_gen_dict[patient_id] = gen_indices.tolist()
    
    # Save results
    print(f"Saving selected time points to {output_path}...")
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    
    if merge_with_data:
        if output_format != 'pkl':
            raise ValueError("merge_with_data=True only supports 'pkl' output format")
        
        # Merge into original data
        print("Merging selected time points into original data...")
        output_data = lab_test_data # We modify in place or copy? In place is fine since we loaded it.
        
        for patient_id, indices in t_gen_dict.items():
            if patient_id in output_data:
                # Store as list of floats/ints, consistent with data_example.py
                output_data[patient_id]['target_image_time_list'] = indices
        
        with open(output_path, 'wb') as f:
            pickle.dump(output_data, f)
        print(f"Saved merged dataset with {len(t_gen_dict)} patients to {output_path}")

    elif output_format == 'pkl':
        with open(output_path, 'wb') as f:
            pickle.dump(t_gen_dict, f)
        print(f"Saved {len(t_gen_dict)} patients' time points to {output_path}")
    elif output_format == 'csv':
        # Convert to DataFrame
        max_len = max(len(indices) for indices in t_gen_dict.values())
        data = {'subject_id': []}
        for i in range(max_len):
            data[f't_gen_{i}'] = []
        
        for patient_id, indices in t_gen_dict.items():
            data['subject_id'].append(patient_id)
            for i in range(max_len):
                if i < len(indices):
                    data[f't_gen_{i}'].append(int(indices[i]))
                else:
                    data[f't_gen_{i}'].append(-1)  # Padding
        
        df = pd.DataFrame(data)
        df.to_csv(output_path, index=False)
        print(f"Saved {len(t_gen_dict)} patients' time points to {output_path}")
    else:
        raise ValueError(f"Unsupported output format: {output_format}")
    
    print("Time point selection completed!")


def run_train_then_select(
    lab_test_data_path: str,
    output_state_path: str,
    output_path: str,
    num_gen_points: int = 3,
    epochs: int = 20,
    learning_rate: float = 1e-3,
    device: str = "cuda",
    hidden_dim: int = 64,
    num_ode_layers: int = 3,
    peak_prominence: float = 0.1,
    peak_distance: int = 1,
    seed: int = 2025,
    max_patients: Optional[int] = None,
    merge_with_data: bool = False,
    output_format: str = "pkl",
    log_prefix: str = "[pipeline]",
    print_each_step: bool = False,
) -> None:
    """
    Train TimePointSelector on ``lab_test_data_path``, save to ``output_state_path``,
    then run ``select_time_points_for_patients`` on the same pickle using those weights.
    """
    print(f"{log_prefix} step 1/2: training -> {output_state_path}", flush=True)
    train_time_point_selector_on_pickle(
        lab_test_data_path=lab_test_data_path,
        output_state_path=output_state_path,
        epochs=epochs,
        learning_rate=learning_rate,
        device=device,
        hidden_dim=hidden_dim,
        num_ode_layers=num_ode_layers,
        seed=seed,
        max_patients=max_patients,
        log_prefix=f"{log_prefix}[train]",
        print_each_step=print_each_step,
    )
    print(f"{log_prefix} step 2/2: selecting -> {output_path}", flush=True)
    select_time_points_for_patients(
        lab_test_data_path=lab_test_data_path,
        output_path=output_path,
        num_gen_points=num_gen_points,
        hidden_dim=hidden_dim,
        num_ode_layers=num_ode_layers,
        peak_prominence=peak_prominence,
        peak_distance=peak_distance,
        device=device,
        output_format=output_format,
        merge_with_data=merge_with_data,
        selector_state_path=output_state_path,
    )
    print(f"{log_prefix} train + select completed.", flush=True)


def run_time_point_selection(
    config: str = None,
    lab_test_data_path: str = None,
    output_path: str = None,
    num_gen_points: int = 3,
    hidden_dim: int = 64,
    num_ode_layers: int = 3,
    peak_prominence: float = 0.1,
    peak_distance: int = 1,
    device: str = 'cpu',
    output_format: str = 'pkl',
    merge_with_data: bool = False,
    selector_state_path: Optional[str] = None,
):
    """
    Function to select generation time points.
    Can be called with specific arguments or a config file.
    """
    # Load config if provided
    if config:
        with open(config, 'r') as f:
            config_data = yaml.safe_load(f)
        
        lab_test_data_path = config_data.get('lab_test_data_path', lab_test_data_path)
        output_path = config_data.get('t_gen_output_path', output_path)
        num_gen_points = config_data.get('num_gen_points', num_gen_points)
        hidden_dim = config_data.get('time_selector_hidden_dim', hidden_dim)
        num_ode_layers = config_data.get('time_selector_num_layers', num_ode_layers)
        peak_prominence = config_data.get('peak_prominence', peak_prominence)
        peak_distance = config_data.get('peak_distance', peak_distance)
        device = config_data.get('device', device)
        output_format = config_data.get('t_gen_output_format', output_format)
        merge_with_data = config_data.get('merge_with_data', merge_with_data)
        selector_state_path = config_data.get('selector_state_path', selector_state_path)
    
    if not lab_test_data_path or not output_path:
        raise ValueError("Must provide either 'config' or both 'lab_test_data_path' and 'output_path'")
    
    # Check device availability
    if device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        device = 'cpu'
    
    select_time_points_for_patients(
        lab_test_data_path=lab_test_data_path,
        output_path=output_path,
        num_gen_points=num_gen_points,
        hidden_dim=hidden_dim,
        num_ode_layers=num_ode_layers,
        peak_prominence=peak_prominence,
        peak_distance=peak_distance,
        device=device,
        output_format=output_format,
        merge_with_data=merge_with_data,
        selector_state_path=selector_state_path,
    )


def main_cli() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Train and/or select generation time points. "
            "Provide --output_state_path to train, --output_path to select; "
            "both together runs train then select (same --lab_test_data_path)."
        )
    )
    parser.add_argument("--lab_test_data_path", type=str, required=True)
    parser.add_argument(
        "--output_state_path",
        type=str,
        default=None,
        help="If set: train TimePointSelector and save weights to this .pt path.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        help="If set: run time-point selection and write results to this path.",
    )
    parser.add_argument("--merge_with_data", action="store_true")
    parser.add_argument("--num_gen_points", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--num_ode_layers", type=int, default=3)
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--max_patients", type=int, default=None)
    parser.add_argument("--verbose_steps", action="store_true", help="Print per-patient loss during training.")
    parser.add_argument(
        "--selector_state_path",
        type=str,
        default=None,
        help="Select-only: load weights from this .pt. Ignored when --output_state_path is set (train+select uses new weights).",
    )
    parser.add_argument("--peak_prominence", type=float, default=0.1)
    parser.add_argument("--peak_distance", type=int, default=1)
    parser.add_argument("--output_format", type=str, default="pkl", choices=("pkl", "csv"))
    args = parser.parse_args()

    if not args.output_state_path and not args.output_path:
        parser.error("Provide at least one of --output_state_path (train) or --output_path (select).")

    if args.output_state_path and args.output_path:
        run_train_then_select(
            lab_test_data_path=args.lab_test_data_path,
            output_state_path=args.output_state_path,
            output_path=args.output_path,
            num_gen_points=args.num_gen_points,
            epochs=args.epochs,
            learning_rate=args.lr,
            device=args.device,
            hidden_dim=args.hidden_dim,
            num_ode_layers=args.num_ode_layers,
            peak_prominence=args.peak_prominence,
            peak_distance=args.peak_distance,
            seed=args.seed,
            max_patients=args.max_patients,
            merge_with_data=args.merge_with_data,
            output_format=args.output_format,
            print_each_step=args.verbose_steps,
        )
    elif args.output_state_path:
        train_time_point_selector_on_pickle(
            lab_test_data_path=args.lab_test_data_path,
            output_state_path=args.output_state_path,
            epochs=args.epochs,
            learning_rate=args.lr,
            device=args.device,
            hidden_dim=args.hidden_dim,
            num_ode_layers=args.num_ode_layers,
            seed=args.seed,
            max_patients=args.max_patients,
            print_each_step=args.verbose_steps,
        )
    else:
        select_time_points_for_patients(
            lab_test_data_path=args.lab_test_data_path,
            output_path=args.output_path,
            num_gen_points=args.num_gen_points,
            hidden_dim=args.hidden_dim,
            num_ode_layers=args.num_ode_layers,
            peak_prominence=args.peak_prominence,
            peak_distance=args.peak_distance,
            device=args.device,
            output_format=args.output_format,
            merge_with_data=args.merge_with_data,
            selector_state_path=args.selector_state_path,
        )


if __name__ == "__main__":
    import sys

    if len(sys.argv) >= 2 and not sys.argv[1].startswith("-"):
        print(
            f"Invalid first argument {sys.argv[1]!r}: this script has no subcommands. "
            "Use flags only, for example:\n"
            "  python utils/select_time_points.py --lab_test_data_path DATA.pkl "
            "--output_state_path sel.pt --output_path OUT.pkl --merge_with_data\n",
            file=sys.stderr,
        )
        sys.exit(2)
    main_cli()

