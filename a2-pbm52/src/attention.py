"""
Attention Mechanisms for Multimodal Fusion

Implements:
1. CrossModalAttention: Attention between different modalities
2. TemporalAttention: Attention over time steps in sequences
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple


class CrossModalAttention(nn.Module):
    """
    Cross-modal attention: Modality A attends to Modality B.
    
    Example: Video features attend to IMU features to incorporate
    relevant motion information at each timestep.
    """
    
    def __init__(
        self,
        query_dim: int,
        key_dim: int,
        hidden_dim: int = 256,
        num_heads: int = 4,
        dropout: float = 0.1
    ):
        """
        Args:
            query_dim: Dimension of query modality features
            key_dim: Dimension of key/value modality features  
            hidden_dim: Hidden dimension for attention computation
            num_heads: Number of attention heads
            dropout: Dropout probability
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        assert hidden_dim % num_heads == 0, \
            f"hidden_dim ({hidden_dim}) must be divisible by num_heads ({num_heads})"
        
        # TODO: Implement multi-head attention projections
        # Hint: Use nn.Linear for Q, K, V projections
        # Query from modality A, Key and Value from modality B
        
        raise NotImplementedError("Implement cross-modal attention projections")
    
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for cross-modal attention.
        
        Args:
            query: (batch_size, query_dim) - features from modality A
            key: (batch_size, key_dim) - features from modality B
            value: (batch_size, key_dim) - features from modality B
            mask: Optional (batch_size,) - binary mask for valid keys
            
        Returns:
            attended_features: (batch_size, hidden_dim) - query attended by key/value
            attention_weights: (batch_size, num_heads, 1, 1) - attention scores
        """
        batch_size = query.size(0)
        
        # TODO: Implement multi-head attention computation
        # Steps:
        #   1. Project query, key, value to (batch, num_heads, seq_len, head_dim)
        #   2. Compute attention scores: Q @ K^T / sqrt(head_dim)
        #   3. Apply mask if provided (set masked positions to -inf before softmax)
        #   4. Apply softmax to get attention weights
        #   5. Apply attention to values: attn_weights @ V
        #   6. Reshape and project back to hidden_dim
        
        raise NotImplementedError("Implement cross-modal attention forward pass")


class TemporalAttention(nn.Module):
    """
    Temporal attention: Attend over sequence of time steps.
    
    Useful for: Variable-length sequences, weighting important timesteps
    """
    
    def __init__(
        self,
        feature_dim: int,
        hidden_dim: int = 256,
        num_heads: int = 4,
        dropout: float = 0.1
    ):
        """
        Args:
            feature_dim: Dimension of input features at each timestep
            hidden_dim: Hidden dimension for attention
            num_heads: Number of attention heads
            dropout: Dropout probability
        """
        super().__init__()
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        # TODO: Implement self-attention over temporal dimension
        # Hint: Similar to CrossModalAttention but Q, K, V from same modality
        
        raise NotImplementedError("Implement temporal attention")
    
    def forward(
        self,
        sequence: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for temporal attention.
        
        Args:
            sequence: (batch_size, seq_len, feature_dim) - temporal sequence
            mask: Optional (batch_size, seq_len) - binary mask for valid timesteps
            
        Returns:
            attended_sequence: (batch_size, seq_len, hidden_dim) - attended features
            attention_weights: (batch_size, num_heads, seq_len, seq_len)
        """
        # TODO: Implement temporal self-attention
        # Steps:
        #   1. Project sequence to Q, K, V
        #   2. Compute self-attention over sequence length
        #   3. Apply mask for variable-length sequences
        #   4. Return attended sequence and weights
        
        raise NotImplementedError("Implement temporal attention forward pass")
    
    def pool_sequence(
        self,
        sequence: torch.Tensor,
        attention_weights: torch.Tensor
    ) -> torch.Tensor:
        """
        Pool sequence to fixed-size representation using attention weights.
        
        Args:
            sequence: (batch_size, seq_len, hidden_dim)
            attention_weights: (batch_size, num_heads, seq_len, seq_len)
            
        Returns:
            pooled: (batch_size, hidden_dim) - fixed-size representation
        """
        # TODO: Implement attention-based pooling
        # Option 1: Weighted average using mean attention weights
        # Option 2: Learn pooling query vector
        # Option 3: Take output at special [CLS] token position
        
        raise NotImplementedError("Implement attention-based pooling")


class PairwiseModalityAttention(nn.Module):
    """
    Pairwise attention between all modality combinations.
    
    For M modalities, computes M*(M-1)/2 pairwise attention operations.
    Example: {video, audio, IMU} -> {video<->audio, video<->IMU, audio<->IMU}
    """
    
    def __init__(
        self,
        modality_dims: dict,
        hidden_dim: int = 256,
        num_heads: int = 4,
        dropout: float = 0.1
    ):
        """
        Args:
            modality_dims: Dict mapping modality names to feature dimensions
                          Example: {'video': 512, 'audio': 128, 'imu': 64}
            hidden_dim: Hidden dimension for attention
            num_heads: Number of attention heads
            dropout: Dropout probability
        """
        super().__init__()
        self.modality_names = list(modality_dims.keys())
        self.num_modalities = len(self.modality_names)
        self.hidden_dim = hidden_dim
        
        # TODO: Create CrossModalAttention for each modality pair
        # Hint: Use nn.ModuleDict with keys like "video_to_audio"
        # For each pair (A, B), create attention A->B and B->A
        
        raise NotImplementedError("Implement pairwise modality attention")
    
    def forward(
        self,
        modality_features: dict,
        modality_mask: Optional[torch.Tensor] = None
    ) -> Tuple[dict, dict]:
        """
        Apply pairwise attention between all modalities.
        
        Args:
            modality_features: Dict of {modality_name: features}
                             Each tensor: (batch_size, feature_dim)
            modality_mask: (batch_size, num_modalities) - availability mask
            
        Returns:
            attended_features: Dict of {modality_name: attended_features}
            attention_maps: Dict of {f"{mod_a}_to_{mod_b}": attention_weights}
        """
        # TODO: Implement pairwise attention
        # Steps:
        #   1. For each modality pair (A, B):
        #      - Apply attention A->B (A attends to B)
        #      - Apply attention B->A (B attends to A)
        #   2. Aggregate attended features (options: sum, concat, gating)
        #   3. Handle missing modalities using mask
        #   4. Return attended features and attention maps for visualization
        
        raise NotImplementedError("Implement pairwise attention forward pass")


def visualize_attention(
    attention_weights: torch.Tensor,
    modality_names: list,
    save_path: str = None
) -> None:
    """
    Visualize attention weights between modalities.
    
    Args:
        attention_weights: (num_heads, num_queries, num_keys) or similar
        modality_names: List of modality names for labeling
        save_path: Optional path to save visualization
    """
    import matplotlib.pyplot as plt
    import numpy as np
    
    # TODO: Implement attention visualization
    # Create heatmap showing which modalities attend to which
    # Useful for understanding fusion behavior
    
    raise NotImplementedError("Implement attention visualization")


if __name__ == '__main__':
    # Simple test
    print("Testing attention mechanisms...")
    
    batch_size = 4
    query_dim = 512  # e.g., video features
    key_dim = 64     # e.g., IMU features
    hidden_dim = 256
    num_heads = 4
    
    # Test CrossModalAttention
    print("\nTesting CrossModalAttention...")
    try:
        attn = CrossModalAttention(query_dim, key_dim, hidden_dim, num_heads)
        
        query = torch.randn(batch_size, query_dim)
        key = torch.randn(batch_size, key_dim)
        value = torch.randn(batch_size, key_dim)
        
        attended, weights = attn(query, key, value)
        
        assert attended.shape == (batch_size, hidden_dim)
        print(f"✓ CrossModalAttention working! Output shape: {attended.shape}")
        
    except NotImplementedError:
        print("✗ CrossModalAttention not implemented yet")
    except Exception as e:
        print(f"✗ CrossModalAttention error: {e}")
    
    # Test TemporalAttention
    print("\nTesting TemporalAttention...")
    try:
        seq_len = 10
        feature_dim = 128
        
        temporal_attn = TemporalAttention(feature_dim, hidden_dim, num_heads)
        sequence = torch.randn(batch_size, seq_len, feature_dim)
        
        attended_seq, weights = temporal_attn(sequence)
        
        assert attended_seq.shape == (batch_size, seq_len, hidden_dim)
        print(f"✓ TemporalAttention working! Output shape: {attended_seq.shape}")
        
    except NotImplementedError:
        print("✗ TemporalAttention not implemented yet")
    except Exception as e:
        print(f"✗ TemporalAttention error: {e}")

