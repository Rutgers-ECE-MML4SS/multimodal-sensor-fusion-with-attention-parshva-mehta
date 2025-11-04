"""
Multimodal Fusion Architectures for Sensor Integration

This module implements three fusion strategies:
1. Early Fusion: Concatenate features before processing
2. Late Fusion: Independent processing, combine predictions
3. Hybrid Fusion: Cross-modal attention + learned weighting
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple

from attention import CrossModalAttention


class EarlyFusion(nn.Module):
    """
    Early fusion: Concatenate encoder outputs and process jointly.
    
    Pros: Joint representation learning across modalities
    Cons: Requires temporal alignment, sensitive to missing modalities
    """
    
    def __init__(
        self,
        modality_dims: Dict[str, int],
        hidden_dim: int = 256,
        num_classes: int = 11,
        dropout: float = 0.1
    ):
        """
        Args:
            modality_dims: Dictionary mapping modality name to feature dimension
                          Example: {'video': 512, 'imu': 64}
            hidden_dim: Hidden dimension for fusion network
            num_classes: Number of output classes
            dropout: Dropout probability
        """
        super().__init__()
        self.modality_names = list(modality_dims.keys())
        
        # TODO: Implement early fusion architecture
        # Hint: Concatenate all modality features, pass through MLP
        # Architecture suggestion:
        #   concat_dim = sum(modality_dims.values())
        #   Linear(concat_dim, hidden_dim) -> ReLU -> Dropout
        #   Linear(hidden_dim, hidden_dim) -> ReLU -> Dropout
        #   Linear(hidden_dim, num_classes)
        
        raise NotImplementedError("Implement early fusion architecture")
    
    def forward(
        self,
        modality_features: Dict[str, torch.Tensor],
        modality_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass with early fusion.
        
        Args:
            modality_features: Dict of {modality_name: features}
                             Each tensor shape: (batch_size, feature_dim)
            modality_mask: Binary mask (batch_size, num_modalities)
                          1 = available, 0 = missing
                          
        Returns:
            logits: (batch_size, num_classes)
        """
        # TODO: Implement forward pass
        # Steps:
        #   1. Extract features for each modality from dict
        #   2. Handle missing modalities (use zeros or learned embeddings)
        #   3. Concatenate all features
        #   4. Pass through fusion network
        
        raise NotImplementedError("Implement early fusion forward pass")


class LateFusion(nn.Module):
    """
    Late fusion: Independent classifiers per modality, combine predictions.
    
    Pros: Handles asynchronous sensors, modular per-modality training
    Cons: Limited cross-modal interaction, fusion only at decision level
    """
    
    def __init__(
        self,
        modality_dims: Dict[str, int],
        hidden_dim: int = 256,
        num_classes: int = 11,
        dropout: float = 0.1
    ):
        """
        Args:
            modality_dims: Dictionary mapping modality name to feature dimension
            hidden_dim: Hidden dimension for per-modality classifiers
            num_classes: Number of output classes
            dropout: Dropout probability
        """
        super().__init__()
        self.modality_names = list(modality_dims.keys())
        self.num_modalities = len(self.modality_names)
        
        # TODO: Create separate classifier for each modality
        # Hint: Use nn.ModuleDict to store per-modality classifiers
        # Each classifier: Linear(modality_dim, hidden_dim) -> ReLU -> Linear(hidden_dim, num_classes)
        
        # TODO: Learn fusion weights (how to combine predictions)
        # Option 1: Learnable weights (nn.Parameter)
        # Option 2: Attention over predictions
        # Option 3: Simple averaging
        
        raise NotImplementedError("Implement late fusion architecture")
    
    def forward(
        self,
        modality_features: Dict[str, torch.Tensor],
        modality_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass with late fusion.
        
        Args:
            modality_features: Dict of {modality_name: features}
            modality_mask: Binary mask for available modalities
            
        Returns:
            logits: (batch_size, num_classes) - fused predictions
            per_modality_logits: Dict of individual modality predictions
        """
        # TODO: Implement forward pass
        # Steps:
        #   1. Get predictions from each modality classifier
        #   2. Handle missing modalities (mask out or skip)
        #   3. Combine predictions using fusion weights
        #   4. Return both fused and per-modality predictions
        
        raise NotImplementedError("Implement late fusion forward pass")


class HybridFusion(nn.Module):
    """
    Hybrid fusion: Cross-modal attention + learned fusion weights.
    
    Pros: Rich cross-modal interaction, robust to missing modalities
    Cons: More complex, higher computation cost
    
    This is the main focus of the assignment!
    """
    
    def __init__(
        self,
        modality_dims: Dict[str, int],
        hidden_dim: int = 256,
        num_classes: int = 11,
        num_heads: int = 4,
        dropout: float = 0.1
    ):
        """
        Args:
            modality_dims: Dictionary mapping modality name to feature dimension
            hidden_dim: Hidden dimension for fusion
            num_classes: Number of output classes
            num_heads: Number of attention heads
            dropout: Dropout probability
        """
        super().__init__()
        self.modality_names = list(modality_dims.keys())
        self.num_modalities = len(self.modality_names)
        self.hidden_dim = hidden_dim
        
        # TODO: Project each modality to common hidden dimension
        # Hint: Use nn.ModuleDict with Linear layers per modality
        
        # TODO: Implement cross-modal attention
        # Use CrossModalAttention from attention.py
        # Each modality should attend to all other modalities
        
        # TODO: Learn adaptive fusion weights based on modality availability
        # Hint: Small MLP that takes modality mask and outputs weights
        
        # TODO: Final classifier
        # Takes fused representation -> num_classes logits
        
        raise NotImplementedError("Implement hybrid fusion architecture")
    
    def forward(
        self,
        modality_features: Dict[str, torch.Tensor],
        modality_mask: Optional[torch.Tensor] = None,
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[Dict]]:
        """
        Forward pass with hybrid fusion.
        
        Args:
            modality_features: Dict of {modality_name: features}
            modality_mask: Binary mask for available modalities
            return_attention: If True, return attention weights for visualization
            
        Returns:
            logits: (batch_size, num_classes)
            attention_info: Optional dict with attention weights and fusion weights
        """
        # TODO: Implement forward pass
        # Steps:
        #   1. Project all modalities to common hidden dimension
        #   2. Apply cross-modal attention between modality pairs
        #   3. Compute adaptive fusion weights based on modality_mask
        #   4. Fuse attended representations with learned weights
        #   5. Pass through final classifier
        #   6. Optionally return attention weights for visualization
        
        raise NotImplementedError("Implement hybrid fusion forward pass")
    
    def compute_adaptive_weights(
        self,
        modality_features: Dict[str, torch.Tensor],
        modality_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute adaptive fusion weights based on modality availability.
        
        Args:
            modality_features: Dict of modality features
            modality_mask: (batch_size, num_modalities) binary mask
            
        Returns:
            weights: (batch_size, num_modalities) normalized fusion weights
        """
        # TODO: Implement adaptive weighting
        # Ideas:
        #   1. Learn weight predictor from modality features + mask
        #   2. Higher weights for more reliable/informative modalities
        #   3. Ensure weights sum to 1 (softmax) and respect mask
        
        raise NotImplementedError("Implement adaptive weight computation")


# Helper functions

def build_fusion_model(
    fusion_type: str,
    modality_dims: Dict[str, int],
    num_classes: int,
    **kwargs
) -> nn.Module:
    """
    Factory function to build fusion models.
    
    Args:
        fusion_type: One of ['early', 'late', 'hybrid']
        modality_dims: Dictionary mapping modality names to dimensions
        num_classes: Number of output classes
        **kwargs: Additional arguments for fusion model
        
    Returns:
        Fusion model instance
    """
    fusion_classes = {
        'early': EarlyFusion,
        'late': LateFusion,
        'hybrid': HybridFusion,
    }
    
    if fusion_type not in fusion_classes:
        raise ValueError(f"Unknown fusion type: {fusion_type}")
    
    return fusion_classes[fusion_type](
        modality_dims=modality_dims,
        num_classes=num_classes,
        **kwargs
    )


if __name__ == '__main__':
    # Simple test to verify implementation
    print("Testing fusion architectures...")
    
    # Test configuration
    modality_dims = {'video': 512, 'imu': 64}
    num_classes = 11
    batch_size = 4
    
    # Create dummy features
    features = {
        'video': torch.randn(batch_size, 512),
        'imu': torch.randn(batch_size, 64)
    }
    mask = torch.tensor([[1, 1], [1, 0], [0, 1], [1, 1]])  # Different availability patterns
    
    # Test each fusion type
    for fusion_type in ['early', 'late', 'hybrid']:
        print(f"\nTesting {fusion_type} fusion...")
        try:
            model = build_fusion_model(fusion_type, modality_dims, num_classes)
            
            if fusion_type == 'late':
                logits, per_mod_logits = model(features, mask)
            else:
                logits = model(features, mask)
            
            assert logits.shape == (batch_size, num_classes), \
                f"Expected shape ({batch_size}, {num_classes}), got {logits.shape}"
            print(f"✓ {fusion_type} fusion working! Output shape: {logits.shape}")
            
        except NotImplementedError:
            print(f"✗ {fusion_type} fusion not implemented yet")
        except Exception as e:
            print(f"✗ {fusion_type} fusion error: {e}")

