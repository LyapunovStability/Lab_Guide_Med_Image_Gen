"""
Standalone script for selecting generation time points T_gen.

This script uses the TimePointSelection module to compute temporal density
and select generation time points for each patient. It ensures constant |T_gen|
across all patients and saves the results for use in training/inference.
"""

import argparse
import os
import pickle
import numpy as np
import pandas as pd
import torch
import yaml
from tqdm import tqdm

from models.TimePointSelector import TimePointSelector


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
    merge_with_data: bool = False
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
    merge_with_data: bool = False
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
        merge_with_data=merge_with_data
    )



