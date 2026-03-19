import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

class GraphAttentionLayer(nn.Module):
    """
    Graph Attention Layer (GAT) for modeling inter-organ dependencies.
    Implements multi-head graph attention mechanism.
    """
    
    def __init__(self, in_features: int, out_features: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_heads = num_heads
        self.head_dim = out_features // num_heads
        
        assert out_features % num_heads == 0, "out_features must be divisible by num_heads"
        
        # Linear transformations for each head
        self.W = nn.Linear(in_features, out_features, bias=False)
        
        # Attention mechanism: a^T [Wh_i || Wh_j]
        self.a = nn.Parameter(torch.empty(size=(2 * self.head_dim, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        
        self.dropout = nn.Dropout(dropout)
        self.leaky_relu = nn.LeakyReLU(0.2)
        
    def forward(self, h: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Apply graph attention.
        
        Args:
            h: (batch_size, num_organs, in_features) organ state features
            mask: (batch_size, num_organs) binary mask indicating valid organs
        
        Returns:
            Updated features: (batch_size, num_organs, out_features)
        """
        batch_size, num_organs, _ = h.shape
        
        # Linear transformation
        Wh = self.W(h)  # (batch_size, num_organs, out_features)
        
        # Reshape for multi-head attention
        Wh = Wh.view(batch_size, num_organs, self.num_heads, self.head_dim)
        Wh = Wh.transpose(1, 2)  # (batch_size, num_heads, num_organs, head_dim)
        
        # Compute attention coefficients
        # For each pair (i, j), compute a^T [Wh_i || Wh_j]
        Wh1 = Wh.unsqueeze(3)  # (batch_size, num_heads, num_organs, 1, head_dim)
        Wh2 = Wh.unsqueeze(2)  # (batch_size, num_heads, 1, num_organs, head_dim)
        
        # Concatenate
        Wh_concat = torch.cat([Wh1.repeat(1, 1, 1, num_organs, 1), 
                              Wh2.repeat(1, 1, num_organs, 1, 1)], dim=-1)
        # (batch_size, num_heads, num_organs, num_organs, 2*head_dim)
        
        # Compute attention scores
        e = torch.matmul(Wh_concat, self.a).squeeze(-1)  # (batch_size, num_heads, num_organs, num_organs)
        e = self.leaky_relu(e)
        
        # Apply mask: set attention to -inf for masked organs.
        # Build pairwise validity matrix with shape:
        # (batch_size, 1, num_organs, num_organs) -> expand across heads.
        node_mask = mask.unsqueeze(1).bool()  # (batch_size, 1, num_organs)
        mask_matrix = node_mask.unsqueeze(-1) & node_mask.unsqueeze(-2)
        mask_matrix = mask_matrix.expand(-1, self.num_heads, -1, -1)
        e = e.masked_fill(~mask_matrix.bool(), float('-inf'))
        
        # Softmax over neighbors
        attention = F.softmax(e, dim=-1)  # (batch_size, num_heads, num_organs, num_organs)
        attention = self.dropout(attention)
        attention = attention.masked_fill(~mask_matrix.bool(), 0.0)
        
        # Apply attention to features
        h_prime = torch.matmul(attention, Wh)  # (batch_size, num_heads, num_organs, head_dim)
        
        # Concatenate heads
        h_prime = h_prime.transpose(1, 2).contiguous()  # (batch_size, num_organs, num_heads, head_dim)
        h_prime = h_prime.view(batch_size, num_organs, self.out_features)  # (batch_size, num_organs, out_features)
        
        # Apply mask
        mask_expanded = mask.unsqueeze(-1)
        h_prime = h_prime * mask_expanded
        
        return h_prime


class InterOrganGNN(nn.Module):
    """
    Graph Attention Network (GAT) for modeling inter-organ dependencies.
    Uses multi-head graph attention layers.
    """
    
    def __init__(self, organ_feat_dim: int, num_layers: int = 2, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.num_layers = num_layers
        self.organ_feat_dim = organ_feat_dim
        
        # Stack of GAT layers
        self.gat_layers = nn.ModuleList()
        for i in range(num_layers):
            in_dim = organ_feat_dim if i == 0 else organ_feat_dim
            self.gat_layers.append(
                GraphAttentionLayer(
                    in_features=in_dim,
                    out_features=organ_feat_dim,
                    num_heads=num_heads,
                    dropout=dropout
                )
            )
        
        self.norm = nn.LayerNorm(organ_feat_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        organ_states: torch.Tensor,
        organ_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Apply GAT to model inter-organ dependencies.
        
        Args:
            organ_states: (batch_size, num_time_steps, num_organs, organ_feat_dim)
            organ_mask: (batch_size, num_time_steps, num_organs) binary mask
        
        Returns:
            Updated organ states: (batch_size, num_time_steps, num_organs, organ_feat_dim)
        """
        batch_size, num_time_steps, num_organs, feat_dim = organ_states.shape
        
        # Process each time step independently
        output_states = []
        
        for t in range(num_time_steps):
            # Get states and mask for this time step
            states_t = organ_states[:, t]  # (batch_size, num_organs, feat_dim)
            mask_t = organ_mask[:, t]  # (batch_size, num_organs)
            
            # Apply GAT layers
            x = states_t
            for i, gat_layer in enumerate(self.gat_layers):
                # Graph attention
                x_new = gat_layer(x, mask_t)
                
                # Residual connection (if dimensions match)
                if i > 0 or x.shape == x_new.shape:
                    x = x + x_new
                else:
                    x = x_new
                
                # Activation and dropout
                x = F.elu(x)
                x = self.dropout(x)
            
            # Layer norm
            x = self.norm(x)
            
            # Apply mask again
            mask_expanded = mask_t.unsqueeze(-1)
            x = x * mask_expanded
            
            output_states.append(x)
        
        # Stack time steps
        output = torch.stack(output_states, dim=1)  # (batch_size, num_time_steps, num_organs, organ_feat_dim)
        
        return output


class TemporalAttention(nn.Module):
    """
    Temporal attention mechanism for modeling temporal dependencies within each organ.
    """
    
    def __init__(self, organ_feat_dim: int, num_heads: int = 4):
        super().__init__()
        self.organ_feat_dim = organ_feat_dim
        self.num_heads = num_heads
        self.head_dim = organ_feat_dim // num_heads
        
        assert organ_feat_dim % num_heads == 0, "organ_feat_dim must be divisible by num_heads"
        
        self.q_proj = nn.Linear(organ_feat_dim, organ_feat_dim)
        self.k_proj = nn.Linear(organ_feat_dim, organ_feat_dim)
        self.v_proj = nn.Linear(organ_feat_dim, organ_feat_dim)
        self.out_proj = nn.Linear(organ_feat_dim, organ_feat_dim)
        
        self.norm = nn.LayerNorm(organ_feat_dim)
    
    def forward(
        self,
        organ_states: torch.Tensor,
        organ_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Apply temporal attention to each organ's state sequence.
        
        Args:
            organ_states: (batch_size, num_time_steps, num_organs, organ_feat_dim)
            organ_mask: (batch_size, num_time_steps, num_organs) binary mask
        
        Returns:
            Updated organ states: (batch_size, num_time_steps, num_organs, organ_feat_dim)
        """
        batch_size, num_time_steps, num_organs, feat_dim = organ_states.shape
        
        # Process each organ independently
        output_states = []
        
        for org_idx in range(num_organs):
            # Get sequence for this organ: (batch_size, num_time_steps, feat_dim)
            org_seq = organ_states[:, :, org_idx, :]
            org_mask_seq = organ_mask[:, :, org_idx]  # (batch_size, num_time_steps)
            
            # Compute Q, K, V
            Q = self.q_proj(org_seq)  # (batch_size, num_time_steps, feat_dim)
            K = self.k_proj(org_seq)
            V = self.v_proj(org_seq)
            
            # Reshape for multi-head attention
            Q = Q.view(batch_size, num_time_steps, self.num_heads, self.head_dim).transpose(1, 2)
            K = K.view(batch_size, num_time_steps, self.num_heads, self.head_dim).transpose(1, 2)
            V = V.view(batch_size, num_time_steps, self.num_heads, self.head_dim).transpose(1, 2)
            # Now shape: (batch_size, num_heads, num_time_steps, head_dim)
            
            # Compute attention scores
            scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)
            # scores: (batch_size, num_heads, num_time_steps, num_time_steps)
            
            # Apply mask: set masked positions to large negative value
            mask_expanded = org_mask_seq.unsqueeze(1).unsqueeze(1)  # (batch_size, 1, 1, num_time_steps)
            mask_expanded = mask_expanded.expand(-1, self.num_heads, num_time_steps, -1)
            scores = scores.masked_fill(~mask_expanded, float('-inf'))
            
            # Apply attention
            attn_weights = F.softmax(scores, dim=-1)
            attn_weights = attn_weights.masked_fill(~mask_expanded, 0.0)
            
            # Apply attention to values
            attended = torch.matmul(attn_weights, V)  # (batch_size, num_heads, num_time_steps, head_dim)
            
            # Reshape back
            attended = attended.transpose(1, 2).contiguous()
            attended = attended.view(batch_size, num_time_steps, feat_dim)
            
            # Output projection
            output = self.out_proj(attended)
            
            # Residual connection and layer norm
            output = self.norm(output + org_seq)
            
            # Apply mask
            mask_expanded = org_mask_seq.unsqueeze(-1)
            output = output * mask_expanded
            
            output_states.append(output)
        
        # Stack organs: (batch_size, num_time_steps, num_organs, feat_dim)
        output = torch.stack(output_states, dim=2)
        
        return output


class TrajectoryEncoder(nn.Module):
    """
    Modal-Shared Trajectory Encoder
    
    Constructs organ state trajectories over time and models information exchange
    among organs through stacked layers of GNN and temporal attention.
    """
    
    def __init__(
        self,
        organ_feat_dim: int,
        num_layers: int = 3,
        gnn_num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.1
    ):
        """
        Initialize trajectory encoder.
        
        Args:
            organ_feat_dim: Dimension of organ state features
            num_layers: Number of stacked layers
            gnn_num_layers: Number of GAT layers within each trajectory layer
            num_heads: Number of attention heads for GAT and temporal attention
            dropout: Dropout rate
        """
        super().__init__()
        
        self.organ_feat_dim = organ_feat_dim
        self.num_layers = num_layers
        
        # Stack of trajectory layers
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            self.layers.append(
                TrajectoryLayer(
                    organ_feat_dim=organ_feat_dim,
                    gnn_num_layers=gnn_num_layers,
                    num_heads=num_heads,
                    dropout=dropout
                )
            )
    
    def forward(
        self,
        organ_states: torch.Tensor,
        organ_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Encode organ state trajectory.
        
        Args:
            organ_states: (batch_size, num_time_steps, num_organs, organ_feat_dim)
            organ_mask: (batch_size, num_time_steps, num_organs) binary mask
        
        Returns:
            Encoded organ states: (batch_size, num_time_steps, num_organs, organ_feat_dim)
        """
        x = organ_states
        
        # Apply stacked layers
        for layer in self.layers:
            x = layer(x, organ_mask)
        
        return x
    
    def infer_at_time(
        self,
        organ_states: torch.Tensor,
        organ_mask: torch.Tensor,
        target_time_indices: torch.Tensor
    ) -> torch.Tensor:
        """
        Infer organ states at specific time points.
        
        Args:
            organ_states: (batch_size, num_time_steps, num_organs, organ_feat_dim)
            organ_mask: (batch_size, num_time_steps, num_organs) binary mask
            target_time_indices: (batch_size, num_target_times) indices of target time points
        
        Returns:
            Inferred organ states: (batch_size, num_target_times, num_organs, organ_feat_dim)
        """
        # Extract states at target time indices
        encoded_states = organ_states
        batch_size = encoded_states.shape[0]
        num_target_times = target_time_indices.shape[1]
        num_organs = encoded_states.shape[2]
        feat_dim = encoded_states.shape[3]
        
        inferred_states = torch.zeros(
            batch_size, num_target_times, num_organs, feat_dim,
            device=encoded_states.device, dtype=encoded_states.dtype
        )
        
        for b in range(batch_size):
            for t_idx, target_t in enumerate(target_time_indices[b]):
                if target_t < encoded_states.shape[1]:
                    inferred_states[b, t_idx] = encoded_states[b, target_t]
        
        return inferred_states


class TrajectoryLayer(nn.Module):
    """
    Single layer of trajectory encoder combining GAT and temporal attention.
    """
    
    def __init__(
        self,
        organ_feat_dim: int,
        gnn_num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.gat = InterOrganGNN(organ_feat_dim, gnn_num_layers, num_heads, dropout)
        self.temporal_attn = TemporalAttention(organ_feat_dim, num_heads)
        
        self.norm1 = nn.LayerNorm(organ_feat_dim)
        self.norm2 = nn.LayerNorm(organ_feat_dim)
    
    def forward(
        self,
        organ_states: torch.Tensor,
        organ_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Apply GAT and temporal attention.
        
        Args:
            organ_states: (batch_size, num_time_steps, num_organs, organ_feat_dim)
            organ_mask: (batch_size, num_time_steps, num_organs) binary mask
        
        Returns:
            Updated organ states: (batch_size, num_time_steps, num_organs, organ_feat_dim)
        """
        # Inter-organ dependency modeling using GAT
        x = self.gat(organ_states, organ_mask)
        x = self.norm1(x + organ_states)  # Residual connection
        
        # Temporal dependency modeling
        x = self.temporal_attn(x, organ_mask)
        x = self.norm2(x + organ_states)  # Residual connection
        
        return x

