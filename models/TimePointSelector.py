import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.signal import find_peaks
from typing import List, Tuple, Optional

class ODENetwork(nn.Module):
    """
    ODE Network for modeling temporal density of laboratory tests.
    """
    
    def __init__(self, hidden_dim: int = 64, num_layers: int = 3):
        super().__init__()
        
        layers = []
        layers.append(nn.Linear(1, hidden_dim))
        layers.append(nn.ReLU())
        
        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
        
        layers.append(nn.Linear(hidden_dim, 1))
        layers.append(nn.Softplus())  # Ensure positive output
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Compute temporal density λ(t) at time points t.
        
        Args:
            t: (batch_size, num_time_points) or (num_time_points,) time points
        
        Returns:
            density: (batch_size, num_time_points) or (num_time_points,) density values
        """
        if t.dim() == 1:
            t = t.unsqueeze(0)  # Add batch dimension
        
        t_normalized = t.unsqueeze(-1)  # (batch_size, num_time_points, 1)
        density = self.network(t_normalized).squeeze(-1)  # (batch_size, num_time_points)
        
        return density


class TimePointSelector(nn.Module):
    """
    Generation Time Point Selection Module
    
    Uses ODE network to model temporal density and peak filtering to identify
    generation time points based on laboratory test sampling density.
    """
    
    def __init__(
        self,
        hidden_dim: int = 64,
        num_ode_layers: int = 3,
        peak_prominence: float = 0.1,
        peak_distance: int = 1
    ):
        """
        Initialize time point selection module.
        
        Args:
            hidden_dim: Hidden dimension for ODE network
            num_ode_layers: Number of layers in ODE network
            peak_prominence: Minimum prominence for peak detection
            peak_distance: Minimum distance between peaks
        """
        super().__init__()
        
        self.ode_network = ODENetwork(hidden_dim, num_ode_layers)
        self.peak_prominence = peak_prominence
        self.peak_distance = peak_distance
    
    def forward(
        self,
        time_points: torch.Tensor,
        lab_test_values: Optional[torch.Tensor] = None,
        num_gen_points: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute temporal density and select generation time points.
        
        Args:
            time_points: (batch_size, num_time_points) time points
            lab_test_values: (batch_size, num_time_points, num_lab_tests) optional lab test values
            num_gen_points: Optional fixed number of generation points. If None, uses variable size.
        
        Returns:
            density: (batch_size, num_time_points) temporal density λ(t)
            gen_time_indices: (batch_size, num_gen_times) indices of selected generation time points
        """
        if num_gen_points is not None:
            return self.select_fixed_number(time_points, num_gen_points)
        
        batch_size, num_time_points = time_points.shape
        
        # Normalize time points to [0, 1] range
        time_min = time_points.min(dim=1, keepdim=True)[0]
        time_max = time_points.max(dim=1, keepdim=True)[0]
        time_range = time_max - time_min
        time_range = time_range.clamp(min=1e-6)  # Avoid division by zero
        time_normalized = (time_points - time_min) / time_range
        
        # Compute temporal density using ODE network
        density = self.ode_network(time_normalized)  # (batch_size, num_time_points)
        
        # Select generation time points using peak filtering
        gen_time_indices_list = []
        
        for b in range(batch_size):
            density_b = density[b].detach().cpu().numpy()
            
            # Find peaks in density curve
            peaks, properties = find_peaks(
                density_b,
                prominence=self.peak_prominence,
                distance=self.peak_distance
            )
            
            # If no peaks found, select time points with highest density
            if len(peaks) == 0:
                # Select top 30% of time points by density
                num_select = max(1, int(0.3 * num_time_points))
                _, top_indices = torch.topk(density[b], k=num_select)
                peaks = top_indices.cpu().numpy()
            
            gen_time_indices_list.append(torch.tensor(peaks, dtype=torch.long))
        
        # Pad to same length (use -1 as padding value)
        max_len = max(len(indices) for indices in gen_time_indices_list)
        gen_time_indices = torch.full(
            (batch_size, max_len),
            -1,
            dtype=torch.long,
            device=time_points.device
        )
        
        for b, indices in enumerate(gen_time_indices_list):
            indices_tensor = indices.to(time_points.device)
            gen_time_indices[b, :len(indices_tensor)] = indices_tensor
        
        return density, gen_time_indices
    
    def select_fixed_number(
        self,
        time_points: torch.Tensor,
        num_gen_points: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Select a fixed number of generation time points for each patient.
        Ensures constant |T_gen| across all patients.
        
        Args:
            time_points: (batch_size, num_time_points) time points
            num_gen_points: Fixed number of generation points (constant across patients)
        
        Returns:
            density: (batch_size, num_time_points) temporal density λ(t)
            gen_time_indices: (batch_size, num_gen_points) indices of selected generation time points
        """
        batch_size, num_time_points = time_points.shape
        
        # Normalize time points to [0, 1] range
        time_min = time_points.min(dim=1, keepdim=True)[0]
        time_max = time_points.max(dim=1, keepdim=True)[0]
        time_range = time_max - time_min
        time_range = time_range.clamp(min=1e-6)  # Avoid division by zero
        time_normalized = (time_points - time_min) / time_range
        
        # Compute temporal density using ODE network
        density = self.ode_network(time_normalized)  # (batch_size, num_time_points)
        
        # Select generation time points with fixed size
        gen_time_indices = torch.zeros(
            (batch_size, num_gen_points),
            dtype=torch.long,
            device=time_points.device
        )
        
        for b in range(batch_size):
            density_b = density[b].detach().cpu().numpy()
            
            # Find peaks in density curve
            peaks, properties = find_peaks(
                density_b,
                prominence=self.peak_prominence,
                distance=self.peak_distance
            )
            
            if len(peaks) >= num_gen_points:
                # If we have more peaks than needed, select top-K by prominence
                if 'prominences' in properties:
                    prominences = properties['prominences']
                    top_k_indices = np.argsort(prominences)[-num_gen_points:]
                    selected_peaks = peaks[top_k_indices]
                else:
                    # Fallback: select top-K by density value
                    peak_densities = density_b[peaks]
                    top_k_indices = np.argsort(peak_densities)[-num_gen_points:]
                    selected_peaks = peaks[top_k_indices]
            else:
                # If we have fewer peaks, select top-K by density
                num_select = min(num_gen_points, num_time_points)
                _, top_indices = torch.topk(density[b], k=num_select)
                selected_peaks = top_indices.cpu().numpy()
            
            # Sort indices for consistency
            selected_peaks = np.sort(selected_peaks)
            
            # Pad if necessary (shouldn't happen, but safety check)
            if len(selected_peaks) < num_gen_points:
                # Repeat last index to pad
                padding = np.full(num_gen_points - len(selected_peaks), selected_peaks[-1] if len(selected_peaks) > 0 else 0)
                selected_peaks = np.concatenate([selected_peaks, padding])
            
            # Take exactly num_gen_points
            selected_peaks = selected_peaks[:num_gen_points]
            
            gen_time_indices[b] = torch.tensor(selected_peaks, dtype=torch.long, device=time_points.device)
        
        return density, gen_time_indices
    
    def compute_loss(
        self,
        time_points: torch.Tensor,
        lab_test_mask: torch.Tensor,
        density: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute loss for ODE network training using inhomogeneous Poisson process framework.
        
        Args:
            time_points: (batch_size, num_time_points) time points
            lab_test_mask: (batch_size, num_time_points) binary mask indicating presence of lab tests
            density: (batch_size, num_time_points) predicted density
        
        Returns:
            loss: Scalar loss value
        """
        # Negative log-likelihood for inhomogeneous Poisson process
        # L = -Σ log(λ(t_i)) + ∫ λ(t) dt
        
        # Compute log-likelihood for observed time points
        mask_expanded = lab_test_mask.unsqueeze(-1)  # (batch_size, num_time_points, 1)
        log_density = torch.log(density + 1e-8)  # Avoid log(0)
        log_likelihood = (log_density * lab_test_mask).sum(dim=1)  # (batch_size,)
        
        # Compute integral ∫ λ(t) dt using trapezoidal rule
        # Normalize time to [0, 1] for integration
        time_min = time_points.min(dim=1, keepdim=True)[0]
        time_max = time_points.max(dim=1, keepdim=True)[0]
        time_range = time_max - time_min
        time_range = time_range.clamp(min=1e-6)
        time_normalized = (time_points - time_min) / time_range
        
        # Sort time points for integration
        sorted_indices = torch.argsort(time_normalized, dim=1)
        sorted_time = torch.gather(time_normalized, 1, sorted_indices)
        sorted_density = torch.gather(density, 1, sorted_indices)
        
        # Trapezoidal integration
        dt = sorted_time[:, 1:] - sorted_time[:, :-1]  # (batch_size, num_time_points-1)
        density_avg = (sorted_density[:, 1:] + sorted_density[:, :-1]) / 2
        integral = (dt * density_avg).sum(dim=1)  # (batch_size,)
        
        # Negative log-likelihood
        loss = -log_likelihood.mean() + integral.mean()
        
        return loss


def select_generation_time_points(
    time_points: torch.Tensor,
    density: torch.Tensor,
    peak_prominence: float = 0.1,
    peak_distance: int = 1
) -> torch.Tensor:
    """
    Select generation time points based on density peaks.
    
    Args:
        time_points: (batch_size, num_time_points) time points
        density: (batch_size, num_time_points) temporal density
        peak_prominence: Minimum prominence for peak detection
        peak_distance: Minimum distance between peaks
    
    Returns:
        gen_time_indices: (batch_size, num_gen_times) indices of selected time points
    """
    batch_size, num_time_points = time_points.shape
    
    gen_time_indices_list = []
    
    for b in range(batch_size):
        density_b = density[b].detach().cpu().numpy()
        
        # Find peaks
        peaks, _ = find_peaks(
            density_b,
            prominence=peak_prominence,
            distance=peak_distance
        )
        
        if len(peaks) == 0:
            # Select top 30% by density
            num_select = max(1, int(0.3 * num_time_points))
            _, top_indices = torch.topk(density[b], k=num_select)
            peaks = top_indices.cpu().numpy()
        
        gen_time_indices_list.append(torch.tensor(peaks, dtype=torch.long))
    
    # Pad to same length
    max_len = max(len(indices) for indices in gen_time_indices_list)
    gen_time_indices = torch.full(
        (batch_size, max_len),
        -1,
        dtype=torch.long,
        device=time_points.device
    )
    
    for b, indices in enumerate(gen_time_indices_list):
        indices_tensor = indices.to(time_points.device)
        gen_time_indices[b, :len(indices_tensor)] = indices_tensor
    
    return gen_time_indices


def compute_t_gen_for_single_patient(
    time_points: torch.Tensor,
    num_gen_points: int,
    hidden_dim: int = 64,
    num_ode_layers: int = 3,
    peak_prominence: float = 0.1,
    peak_distance: int = 1,
    device: str = 'cpu'
) -> List[int]:
    """
    Standalone function to compute T_gen for a single patient.
    Ensures constant |T_gen| = num_gen_points.
    
    This function can be called independently without instantiating the full module.
    Useful for pre-computing T_gen offline.
    
    Args:
        time_points: (num_time_points,) time points for a single patient
        num_gen_points: Fixed number of generation points (constant)
        hidden_dim: Hidden dimension for ODE network
        num_ode_layers: Number of layers in ODE network
        peak_prominence: Minimum prominence for peak detection
        peak_distance: Minimum distance between peaks
        device: Device to run computation on
    
    Returns:
        gen_indices: List of num_gen_points time indices (constant size)
    """
    # Ensure time_points is a tensor
    if not isinstance(time_points, torch.Tensor):
        time_points = torch.tensor(time_points, dtype=torch.float32)
    
    # Add batch dimension if needed
    if time_points.dim() == 1:
        time_points = time_points.unsqueeze(0)
    
    # Move to device
    time_points = time_points.to(device)
    
    # Create time point selector
    selector = TimePointSelector(
        hidden_dim=hidden_dim,
        num_ode_layers=num_ode_layers,
        peak_prominence=peak_prominence,
        peak_distance=peak_distance
    ).to(device)
    selector.eval()
    
    # Compute with fixed number
    with torch.no_grad():
        density, gen_time_indices = selector.select_fixed_number(
            time_points=time_points,
            num_gen_points=num_gen_points
        )
    
    # Extract indices (remove batch dimension)
    gen_indices = gen_time_indices[0].cpu().numpy()
    
    # Remove padding and ensure exact size
    gen_indices = gen_indices[gen_indices >= 0]
    
    # If we have fewer than num_gen_points, select top-K by density
    if len(gen_indices) < num_gen_points:
        density_1d = density[0].cpu().numpy()
        num_select = min(num_gen_points, len(density_1d))
        _, top_indices = torch.topk(
            torch.from_numpy(density_1d),
            k=num_select
        )
        gen_indices = top_indices.numpy()
    
    # Ensure exactly num_gen_points
    gen_indices = gen_indices[:num_gen_points]
    
    # Sort for consistency
    gen_indices = np.sort(gen_indices)
    
    # Validate constant size
    if len(gen_indices) != num_gen_points:
        raise ValueError(
            f"Computed {len(gen_indices)} time points, expected {num_gen_points}"
        )
    
    return gen_indices.tolist()
