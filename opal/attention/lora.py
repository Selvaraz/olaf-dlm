"""
LoRA (Low-Rank Adaptation) Implementation for Opal GPT Model

This module provides a minimal, dependency-light LoRA implementation that can be used
as a fallback when PEFT is not available. The implementation follows the original
LoRA paper (Hu et al., 2021) and provides efficient parameter-efficient fine-tuning.

Key Design Choices:
- Minimal dependencies (only PyTorch)
- Memory-efficient implementation with proper dtype/device handling
- Support for merging and unloading adapters
- Compatible with mixed precision training
- Thread-safe operations for DDP training

LoRA Domain Adaptation: This file contains the core LoRA module implementation
for parameter-efficient domain adaptation in phase-2 training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Union
import math


class LoRAInjectedLinear(nn.Module):
    """
    LoRA Domain Adaptation: Linear layer with LoRA adapters injected.
    
    This class wraps a standard Linear layer and adds low-rank adaptation matrices
    that can be trained while keeping the original weights frozen. The forward pass
    computes: output = base_linear(x) + (x @ A.T @ B.T) * (alpha / rank)
    
    Args:
        base_linear: Original nn.Linear layer to wrap
        rank: LoRA rank (r) - dimensionality of the adaptation matrices
        alpha: LoRA scaling factor (alpha) - typically 2*rank for balanced scaling
        dropout: Dropout probability for LoRA layers
        dtype: Data type for LoRA parameters (defaults to base_linear dtype)
        device: Device for LoRA parameters (defaults to base_linear device)
    """
    
    def __init__(
        self,
        base_linear: nn.Linear,
        rank: int = 16,
        alpha: float = 32.0,
        dropout: float = 0.1,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        
        # LoRA Domain Adaptation: Store reference to base linear layer
        self.base_linear = base_linear
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank  # LoRA Domain Adaptation: Scaling factor for adapter output
        
        # LoRA Domain Adaptation: Determine dtype and device from base layer
        if dtype is None:
            dtype = next(base_linear.parameters()).dtype
        if device is None:
            device = next(base_linear.parameters()).device
            
        # LoRA Domain Adaptation: LoRA adapter matrices A and B
        # Following LoRA paper: W + BA where A is (rank, in_features), B is (out_features, rank)
        self.lora_A = nn.Parameter(
            torch.zeros(rank, base_linear.in_features, dtype=dtype, device=device)
        )
        self.lora_B = nn.Parameter(
            torch.zeros(base_linear.out_features, rank, dtype=dtype, device=device)
        )
        
        # LoRA Domain Adaptation: Dropout for LoRA path
        self.lora_dropout = nn.Dropout(p=dropout) if dropout > 0.0 else nn.Identity()
        
        # LoRA Domain Adaptation: Flag to track if adapters are merged into base weights
        self.merged = False
        
        # LoRA Domain Adaptation: Initialize LoRA weights following the paper
        self.reset_lora_parameters()
        
    def reset_lora_parameters(self):
        """LoRA Domain Adaptation: Initialize LoRA parameters following the original paper."""
        # LoRA Domain Adaptation: A is initialized with Kaiming uniform, B with zeros
        # This ensures the adapter starts with zero output (no interference with base model)
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        LoRA Domain Adaptation: Forward pass with LoRA adaptation.
        
        Computes: output = base_linear(x) + lora_adaptation(x) * scaling
        where lora_adaptation(x) = x @ A.T @ B.T
        """
        # LoRA Domain Adaptation: Always compute base linear output
        base_out = self.base_linear(x)
        
        if self.merged:
            # LoRA Domain Adaptation: If merged, adapters are already in base weights
            return base_out
            
        # LoRA Domain Adaptation: Compute LoRA adaptation path
        # x @ A.T -> (B, T, rank), then @ B.T -> (B, T, out_features)
        lora_out = self.lora_dropout(x) @ self.lora_A.t() @ self.lora_B.t()
        
        # LoRA Domain Adaptation: Apply scaling and add to base output
        return base_out + lora_out * self.scaling
        
    def merge_adapters(self):
        """
        LoRA Domain Adaptation: Merge LoRA adapters into base weights.
        
        After merging, the forward pass will only use the base linear layer
        with the adapter weights incorporated. This is useful for inference
        or when saving a unified model checkpoint.
        """
        if not self.merged:
            # LoRA Domain Adaptation: Compute adapter weight contribution: B @ A
            adapter_weight = self.lora_B @ self.lora_A  # (out_features, in_features)
            
            # LoRA Domain Adaptation: Add scaled adapter weights to base weights
            with torch.no_grad():
                self.base_linear.weight.data += adapter_weight * self.scaling
                
            self.merged = True
            
    def unload_adapters(self):
        """
        LoRA Domain Adaptation: Remove LoRA adapters from base weights.
        
        This reverses the merge operation, restoring the base weights to their
        original state. Useful for switching between merged and unmerged states.
        """
        if self.merged:
            # LoRA Domain Adaptation: Compute adapter weight contribution: B @ A
            adapter_weight = self.lora_B @ self.lora_A  # (out_features, in_features)
            
            # LoRA Domain Adaptation: Subtract scaled adapter weights from base weights
            with torch.no_grad():
                self.base_linear.weight.data -= adapter_weight * self.scaling
                
            self.merged = False
            
    def get_lora_parameters(self):
        """LoRA Domain Adaptation: Get all LoRA parameters for optimizer."""
        return [self.lora_A, self.lora_B]
        
    def get_lora_state_dict(self):
        """LoRA Domain Adaptation: Get LoRA adapter state dict for saving."""
        return {
            'lora_A': self.lora_A.data.clone(),
            'lora_B': self.lora_B.data.clone(),
            'rank': self.rank,
            'alpha': self.alpha,
            'scaling': self.scaling,
            'merged': self.merged,
        }
        
    def load_lora_state_dict(self, state_dict):
        """LoRA Domain Adaptation: Load LoRA adapter state dict."""
        self.lora_A.data.copy_(state_dict['lora_A'])
        self.lora_B.data.copy_(state_dict['lora_B'])
        self.rank = state_dict['rank']
        self.alpha = state_dict['alpha']
        self.scaling = state_dict['scaling']
        self.merged = state_dict.get('merged', False)
        
    def extra_repr(self) -> str:
        """LoRA Domain Adaptation: String representation for debugging."""
        return f'rank={self.rank}, alpha={self.alpha}, scaling={self.scaling:.3f}, merged={self.merged}'


def inject_lora_into_linear(
    linear_layer: nn.Linear,
    rank: int = 16,
    alpha: float = 32.0,
    dropout: float = 0.1,
    dtype: Optional[torch.dtype] = None,
    device: Optional[torch.device] = None,
) -> LoRAInjectedLinear:
    """
    LoRA Domain Adaptation: Inject LoRA adapters into a linear layer.
    
    This function wraps a standard Linear layer with LoRA adapters, allowing
    for parameter-efficient fine-tuning while keeping the original weights frozen.
    
    Args:
        linear_layer: The nn.Linear layer to inject LoRA into
        rank: LoRA rank (r) - controls adapter capacity
        alpha: LoRA scaling factor (alpha) - typically 2*rank
        dropout: Dropout probability for LoRA layers
        dtype: Data type for LoRA parameters
        device: Device for LoRA parameters
        
    Returns:
        LoRAInjectedLinear: The wrapped layer with LoRA adapters
    """
    # LoRA Domain Adaptation: Create LoRA-injected version of the linear layer
    return LoRAInjectedLinear(
        base_linear=linear_layer,
        rank=rank,
        alpha=alpha,
        dropout=dropout,
        dtype=dtype,
        device=device,
    )


def check_peft_availability() -> bool:
    """
    LoRA Domain Adaptation: Check if PEFT library is available.
    
    This function checks for the presence of the PEFT library, which provides
    a more feature-complete LoRA implementation. If PEFT is available, it's
    recommended to use it instead of this minimal fallback implementation.
    
    Returns:
        bool: True if PEFT is available, False otherwise
    """
    try:
        import peft  # LoRA Domain Adaptation: Try importing PEFT
        return True
    except ImportError:
        return False


def get_lora_implementation_info() -> str:
    """
    LoRA Domain Adaptation: Get information about the LoRA implementation being used.
    
    Returns:
        str: Information about the LoRA implementation
    """
    if check_peft_availability():
        return "PEFT library available (recommended for production use)"
    else:
        return "Using minimal fallback LoRA implementation (dependency-light)"
