"""
LoRA Utilities for Opal GPT Model

This module provides utility functions and configuration classes for managing
LoRA (Low-Rank Adaptation) in the Opal GPT model. It includes functions for
injecting and removing LoRA adapters, freezing base model weights, and managing
LoRA parameters during training.

LoRA Domain Adaptation: This file contains utilities for managing LoRA adapters
in the domain adaptation phase, including model injection, parameter management,
and checkpointing support.
"""

import torch
import torch.nn as nn
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple, Union
import json
import hashlib
from pathlib import Path
import warnings
import hashlib
from pathlib import Path

# LoRA Domain Adaptation: Import safetensors with fallback
try:
    import safetensors.torch
    HAS_SAFETENSORS = True
except ImportError:
    HAS_SAFETENSORS = False
    warnings.warn("safetensors not available, falling back to PyTorch format")

from .lora import LoRAInjectedLinear, inject_lora_into_linear


@dataclass
class LoRAConfig:
    """
    LoRA Domain Adaptation: Configuration class for LoRA parameters.
    
    This dataclass encapsulates all LoRA-related configuration parameters,
    providing a clean interface for managing LoRA settings across different
    training phases.
    
    Args:
        use_lora: Whether to enable LoRA adapters
        rank: LoRA rank (r) - controls adapter capacity
        alpha: LoRA scaling factor (alpha) - typically 2*rank
        dropout: Dropout probability for LoRA layers
        target_modules: List of module names to inject LoRA into
        lora_include_mlp: Whether to include MLP layers in LoRA injection
        merge_on_finalize: Whether to merge adapters on training completion
        save_adapters: Whether to save separate adapter weights
        checkpoint_format: Format for adapter checkpoints ('safetensors' or 'pytorch')
    """
    use_lora: bool = False
    rank: int = 16
    alpha: float = 32.0
    dropout: float = 0.1
    target_modules: List[str] = field(default_factory=lambda: ["Wq", "Wk", "Wv", "out_proj"])
    lora_include_mlp: bool = False
    merge_on_finalize: bool = True
    save_adapters: bool = True
    checkpoint_format: str = "safetensors"  # 'safetensors' or 'pytorch'
    
    def __post_init__(self):
        """LoRA Domain Adaptation: Validate configuration parameters."""
        if self.rank <= 0:
            raise ValueError(f"LoRA rank must be positive, got {self.rank}")
        if self.alpha <= 0:
            raise ValueError(f"LoRA alpha must be positive, got {self.alpha}")
        if not 0 <= self.dropout <= 1:
            raise ValueError(f"LoRA dropout must be in [0, 1], got {self.dropout}")
        if self.checkpoint_format not in ["safetensors", "pytorch"]:
            raise ValueError(f"Checkpoint format must be 'safetensors' or 'pytorch', got {self.checkpoint_format}")
            
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'LoRAConfig':
        """LoRA Domain Adaptation: Create LoRAConfig from dictionary."""
        # LoRA Domain Adaptation: Extract LoRA-specific keys and handle nested export config
        lora_keys = {
            'use_lora', 'rank', 'alpha', 'dropout', 'target_modules', 
            'lora_include_mlp', 'merge_on_finalize', 'save_adapters', 'checkpoint_format'
        }
        
        # LoRA Domain Adaptation: Map config keys to LoRAConfig parameters
        lora_config = {}
        for key in lora_keys:
            if key in config_dict:
                lora_config[key] = config_dict[key]
                
        # LoRA Domain Adaptation: Handle nested export configuration
        if 'export' in config_dict:
            export_config = config_dict['export']
            if 'merge_on_finalize' in export_config:
                lora_config['merge_on_finalize'] = export_config['merge_on_finalize']
            if 'save_adapters' in export_config:
                lora_config['save_adapters'] = export_config['save_adapters']
            if 'checkpoint_format' in export_config:
                lora_config['checkpoint_format'] = export_config['checkpoint_format']
                
        # LoRA Domain Adaptation: Handle alternative parameter names
        if 'lora_rank' in config_dict:
            lora_config['rank'] = config_dict['lora_rank']
        if 'lora_alpha' in config_dict:
            lora_config['alpha'] = config_dict['lora_alpha']
        if 'lora_dropout' in config_dict:
            lora_config['dropout'] = config_dict['lora_dropout']
            
        return cls(**lora_config)
        
    def to_dict(self) -> Dict[str, Any]:
        """LoRA Domain Adaptation: Convert LoRAConfig to dictionary."""
        return {
            'use_lora': self.use_lora,
            'rank': self.rank,
            'alpha': self.alpha,
            'dropout': self.dropout,
            'target_modules': self.target_modules,
            'lora_include_mlp': self.lora_include_mlp,
            'merge_on_finalize': self.merge_on_finalize,
            'save_adapters': self.save_adapters,
            'checkpoint_format': self.checkpoint_format,
        }


def inject_lora_into_model(
    model: nn.Module, 
    lora_config: LoRAConfig,
    verbose: bool = True
) -> Tuple[nn.Module, List[str]]:
    """
    LoRA Domain Adaptation: Inject LoRA adapters into specified modules of a model.
    
    This function recursively searches through the model and injects LoRA adapters
    into linear layers whose names match the target_modules specification.
    
    Args:
        model: The model to inject LoRA into
        lora_config: LoRA configuration parameters
        verbose: Whether to print injection information
        
    Returns:
        Tuple[nn.Module, List[str]]: Modified model and list of injected module names
    """
    if not lora_config.use_lora:
        if verbose:
            print("LoRA Domain Adaptation: LoRA disabled, skipping injection")
        return model, []
        
    injected_modules = []  # LoRA Domain Adaptation: Track injected modules
    
    # LoRA Domain Adaptation: Recursively inject LoRA into target modules
    def _inject_lora_recursive(module: nn.Module, name_prefix: str = ""):
        for name, child in module.named_children():
            full_name = f"{name_prefix}.{name}" if name_prefix else name
            
            # LoRA Domain Adaptation: Check if this module should have LoRA injected
            should_inject = False
            
            # LoRA Domain Adaptation: Check against target modules
            for target in lora_config.target_modules:
                if name == target or full_name.endswith(f".{target}"):
                    should_inject = True
                    break
                    
            # LoRA Domain Adaptation: Handle MLP injection if enabled
            if lora_config.lora_include_mlp and isinstance(child, nn.Linear):
                # LoRA Domain Adaptation: Include MLP linear layers (both attention and feedforward)
                if any(mlp_name in full_name.lower() for mlp_name in ['feedforward', 'mlp', 'fc', 'w1', 'w2', 'w3']):
                    should_inject = True
                    
            # LoRA Domain Adaptation: Inject LoRA if conditions are met
            if should_inject and isinstance(child, nn.Linear):
                # LoRA Domain Adaptation: Create LoRA-injected version
                lora_layer = inject_lora_into_linear(
                    linear_layer=child,
                    rank=lora_config.rank,
                    alpha=lora_config.alpha,
                    dropout=lora_config.dropout,
                )
                
                # LoRA Domain Adaptation: Replace the module
                setattr(module, name, lora_layer)
                injected_modules.append(full_name)
                
                if verbose:
                    print(f"LoRA Domain Adaptation: Injected LoRA into {full_name} "
                          f"(rank={lora_config.rank}, alpha={lora_config.alpha})")
            else:
                # LoRA Domain Adaptation: Recursively process child modules
                _inject_lora_recursive(child, full_name)
                
    # LoRA Domain Adaptation: Start recursive injection
    _inject_lora_recursive(model)
    
    if verbose:
        print(f"LoRA Domain Adaptation: Successfully injected LoRA into {len(injected_modules)} modules")
        
    return model, injected_modules


def freeze_base_model_weights(model: nn.Module, verbose: bool = True) -> int:
    """
    LoRA Domain Adaptation: Freeze all non-LoRA parameters in the model.
    
    This function freezes the base model weights while keeping LoRA adapter
    parameters trainable. This is essential for LoRA training where only
    the adapter weights should be updated.
    
    Args:
        model: The model with LoRA adapters injected
        verbose: Whether to print freezing information
        
    Returns:
        int: Number of frozen parameters
    """
    frozen_count = 0
    trainable_count = 0
    lora_param_count = 0
    
    # LoRA Domain Adaptation: Freeze all non-LoRA parameters
    for name, param in model.named_parameters():
        if 'lora_A' in name or 'lora_B' in name:
            # LoRA Domain Adaptation: Keep LoRA parameters trainable
            param.requires_grad = True
            trainable_count += param.numel()
            lora_param_count += param.numel()
            if verbose:
                print(f"LoRA Domain Adaptation: Keeping trainable: {name} ({param.numel()} params)")
        else:
            # LoRA Domain Adaptation: Freeze base model parameters
            param.requires_grad = False
            frozen_count += param.numel()
    
    # LoRA Domain Adaptation: Validate that LoRA parameters were found
    if lora_param_count == 0:
        raise RuntimeError(
            "LoRA Domain Adaptation: No LoRA parameters found! "
            "This indicates LoRA injection failed or was not performed. "
            "Please check that inject_lora_into_model() was called successfully."
        )
            
    if verbose:
        total_params = frozen_count + trainable_count
        lora_percentage = (trainable_count / total_params) * 100 if total_params > 0 else 0
        print(f"LoRA Domain Adaptation: Frozen {frozen_count:,} base parameters")
        print(f"LoRA Domain Adaptation: Trainable {trainable_count:,} LoRA parameters ({lora_percentage:.2f}%)")
        
    return frozen_count


def ensure_lora_device_consistency(model: nn.Module, device: torch.device, verbose: bool = True) -> None:
    """
    LoRA Domain Adaptation: Ensure all LoRA parameters are on the correct device.
    
    This function should be called after model.to(device) to ensure LoRA adapters
    are properly moved to the target device. This is important because LoRA injection
    happens during __init__ and device moves can happen afterwards.
    
    Args:
        model: The model with LoRA adapters injected
        device: Target device for the model
        verbose: Whether to print device move information
    """
    moved_count = 0
    
    for name, module in model.named_modules():
        if isinstance(module, LoRAInjectedLinear):
            # Check if LoRA parameters are on the wrong device
            if module.lora_A.device != device or module.lora_B.device != device:
                module.lora_A.data = module.lora_A.data.to(device)
                module.lora_B.data = module.lora_B.data.to(device)
                moved_count += 1
                if verbose:
                    print(f"LoRA Domain Adaptation: Moved {name} adapters to {device}")
    
    if verbose and moved_count > 0:
        print(f"LoRA Domain Adaptation: Moved {moved_count} adapter modules to {device}")
    elif verbose:
        print(f"LoRA Domain Adaptation: All adapters already on {device}")

def get_lora_parameters(model: nn.Module) -> List[torch.nn.Parameter]:
    """
    LoRA Domain Adaptation: Get all LoRA parameters from the model.
    
    This function collects all LoRA adapter parameters (lora_A and lora_B)
    from the model, which can be used for creating optimizers that only
    update the LoRA parameters.
    
    Args:
        model: The model with LoRA adapters injected
        
    Returns:
        List[torch.nn.Parameter]: List of all LoRA parameters that require gradients
    """
    lora_params = []
    
    # LoRA Domain Adaptation: Collect all LoRA parameters that require gradients
    for name, param in model.named_parameters():
        if ('lora_A' in name or 'lora_B' in name) and param.requires_grad:
            lora_params.append(param)
            
    return lora_params


def merge_lora_weights(model: nn.Module, verbose: bool = True) -> nn.Module:
    """
    LoRA Domain Adaptation: Merge LoRA adapter weights into base model weights.
    
    🔧 CRITICAL FIX: Converts LoRAInjectedLinear back to standard nn.Linear
    with merged weights for proper checkpoint structure.
    """
    import copy
    
    if verbose:
        print("🎯 LoRA: Creating deep copy of model before merge...")
    
    model_copy = copy.deepcopy(model)
    
    merged_count = 0
    replaced_count = 0
    
    # LoRA Domain Adaptation: Merge all LoRA adapters in the COPY and replace with nn.Linear
    for name, parent_module in list(model_copy.named_modules()):  # 🔧 FIX: Use list() to avoid iterator issues
        # Check each child of this module
        for child_name, child_module in list(parent_module.named_children()):  # 🔧 FIX: Use list()
            if isinstance(child_module, LoRAInjectedLinear):
                # 🔧 CRITICAL: First merge the adapters
                child_module.merge_adapters()
                merged_count += 1
                
                # 🔧 CRITICAL: Extract merged weights from base_linear
                merged_weight = child_module.base_linear.weight.data.clone()
                merged_bias = child_module.base_linear.bias.data.clone() if child_module.base_linear.bias is not None else None
                
                # 🔧 CRITICAL: Create standard nn.Linear with merged weights
                in_features = child_module.base_linear.in_features
                out_features = child_module.base_linear.out_features
                has_bias = child_module.base_linear.bias is not None
                
                # Create new standard linear layer
                merged_linear = nn.Linear(in_features, out_features, bias=has_bias)
                merged_linear.weight.data = merged_weight
                if has_bias:
                    merged_linear.bias.data = merged_bias
                
                # 🔧 CRITICAL: Replace LoRAInjectedLinear with standard nn.Linear
                setattr(parent_module, child_name, merged_linear)
                replaced_count += 1
                
                if verbose:
                    print(f"LoRA Domain Adaptation: Merged and replaced {name}.{child_name}")
    
    if verbose:
        print(f"LoRA Domain Adaptation: Successfully merged {merged_count} LoRA adapters")
        print(f"LoRA Domain Adaptation: Replaced {replaced_count} LoRAInjectedLinear with nn.Linear")
        print(f"🎯 LoRA: Converted to standard nn.Linear layers (no LoRA structure)")
        
        # 🔧 VERIFICATION: Check the merged model has NO LoRA structure
        lora_modules_remaining = sum(1 for m in model_copy.modules() if isinstance(m, LoRAInjectedLinear))
        if lora_modules_remaining > 0:
            print(f"❌ WARNING: {lora_modules_remaining} LoRAInjectedLinear modules still in model!")
        else:
            print(f"✅ VERIFIED: No LoRAInjectedLinear modules in merged model")
        
        print(f"🎯 LoRA: Original model unchanged, returning merged copy")
        
    return model_copy


def unload_lora_weights(model: nn.Module, verbose: bool = True) -> nn.Module:
    """
    LoRA Domain Adaptation: Unload LoRA adapter weights from base model weights.
    
    This function reverses the merge operation, restoring the base model weights
    to their original state and reactivating the LoRA computation path.
    
    Args:
        model: The model with merged LoRA adapters
        verbose: Whether to print unloading information
        
    Returns:
        nn.Module: The model with unloaded adapters
    """
    unloaded_count = 0
    
    # LoRA Domain Adaptation: Unload all LoRA adapters
    for name, module in model.named_modules():
        if isinstance(module, LoRAInjectedLinear):
            module.unload_adapters()
            unloaded_count += 1
            if verbose:
                print(f"LoRA Domain Adaptation: Unloaded adapters in {name}")
                
    if verbose:
        print(f"LoRA Domain Adaptation: Successfully unloaded {unloaded_count} LoRA adapters")
        
    return model


def save_lora_adapters(
    model: nn.Module,
    save_path: Union[str, Path],
    lora_config: LoRAConfig,
    base_model_info: Optional[Dict[str, Any]] = None,
    format: str = "safetensors"
) -> Dict[str, Any]:
    """
    LoRA Domain Adaptation: Save LoRA adapter weights and configuration.
    
    This function saves only the LoRA adapter parameters along with metadata
    including configuration, base model information, and verification hashes.
    
    Args:
        model: The model with LoRA adapters
        save_path: Path to save the adapter weights
        lora_config: LoRA configuration used
        base_model_info: Optional information about the base model
        format: Save format ('safetensors' or 'pytorch')
        
    Returns:
        Dict[str, Any]: Manifest information about the saved adapters
    """
    save_path = Path(save_path)
    
    # LoRA Domain Adaptation: Collect LoRA adapter state dicts
    adapter_state_dict = {}
    adapter_info = {}
    
    for name, module in model.named_modules():
        if isinstance(module, LoRAInjectedLinear):
            lora_state = module.get_lora_state_dict()
            adapter_state_dict[f"{name}.lora_A"] = lora_state['lora_A']
            adapter_state_dict[f"{name}.lora_B"] = lora_state['lora_B']
            
            # LoRA Domain Adaptation: Store adapter metadata
            adapter_info[name] = {
                'rank': lora_state['rank'],
                'alpha': lora_state['alpha'],
                'scaling': lora_state['scaling'],
                'merged': lora_state['merged'],
                'shape_A': list(lora_state['lora_A'].shape),
                'shape_B': list(lora_state['lora_B'].shape),
                'dtype': str(lora_state['lora_A'].dtype),
            }
    
    # LoRA Domain Adaptation: Save adapter weights
    if format == "safetensors" and HAS_SAFETENSORS:
        safetensors.torch.save_file(adapter_state_dict, str(save_path.with_suffix('.safetensors')))
    else:
        if format == "safetensors" and not HAS_SAFETENSORS:
            print("LoRA Domain Adaptation: safetensors not available, using PyTorch format")
        torch.save(adapter_state_dict, str(save_path.with_suffix('.pt')))
    
    # LoRA Domain Adaptation: Create manifest with metadata
    manifest = {
        'lora_config': lora_config.to_dict(),
        'adapter_info': adapter_info,
        'base_model_info': base_model_info or {},
        'format': format,
        'total_adapters': len(adapter_info),
        'total_parameters': sum(info['shape_A'][0] * info['shape_A'][1] + 
                                info['shape_B'][0] * info['shape_B'][1] 
                                for info in adapter_info.values()),
        'created_at': torch.utils.data.get_worker_info().id if torch.utils.data.get_worker_info() else 0,
        'pytorch_version': torch.__version__,
    }
    
    # LoRA Domain Adaptation: Add content hash for verification
    if format == "safetensors" and HAS_SAFETENSORS and save_path.with_suffix('.safetensors').exists():
        with open(save_path.with_suffix('.safetensors'), 'rb') as f:
            manifest['content_hash'] = hashlib.sha256(f.read()).hexdigest()
    elif save_path.with_suffix('.pt').exists():
        with open(save_path.with_suffix('.pt'), 'rb') as f:
            manifest['content_hash'] = hashlib.sha256(f.read()).hexdigest()
    
    # LoRA Domain Adaptation: Save manifest
    with open(save_path.with_suffix('.json'), 'w') as f:
        json.dump(manifest, f, indent=2)
    
    return manifest


def load_lora_adapters(
    model: nn.Module,
    load_path: Union[str, Path],
    strict: bool = True
) -> Dict[str, Any]:
    """
    LoRA Domain Adaptation: Load LoRA adapter weights into a model.
    
    This function loads previously saved LoRA adapter weights and applies
    them to the corresponding modules in the model.
    
    Args:
        model: The model to load adapters into (must have LoRA injected)
        load_path: Path to the saved adapter weights
        strict: Whether to require exact module name matching
        
    Returns:
        Dict[str, Any]: Loaded manifest information
    """
    load_path = Path(load_path)
    
    # LoRA Domain Adaptation: Load manifest
    manifest_path = load_path.with_suffix('.json')
    if not manifest_path.exists():
        raise FileNotFoundError(f"LoRA manifest not found: {manifest_path}")
        
    with open(manifest_path, 'r') as f:
        manifest = json.load(f)
    
    # LoRA Domain Adaptation: Determine format and load weights
    format = manifest.get('format', 'pytorch')
    
    if format == "safetensors" and HAS_SAFETENSORS:
        weights_path = load_path.with_suffix('.safetensors')
        if not weights_path.exists():
            raise FileNotFoundError(f"LoRA weights not found: {weights_path}")
        adapter_state_dict = safetensors.torch.load_file(str(weights_path))
    else:
        weights_path = load_path.with_suffix('.pt')
        if not weights_path.exists():
            raise FileNotFoundError(f"LoRA weights not found: {weights_path}")
        adapter_state_dict = torch.load(str(weights_path))
    
    # LoRA Domain Adaptation: Load weights into model
    loaded_modules = []
    for name, module in model.named_modules():
        if isinstance(module, LoRAInjectedLinear):
            lora_A_key = f"{name}.lora_A"
            lora_B_key = f"{name}.lora_B"
            
            if lora_A_key in adapter_state_dict and lora_B_key in adapter_state_dict:
                # LoRA Domain Adaptation: Load adapter weights
                lora_state = {
                    'lora_A': adapter_state_dict[lora_A_key],
                    'lora_B': adapter_state_dict[lora_B_key],
                    'rank': manifest['adapter_info'][name]['rank'],
                    'alpha': manifest['adapter_info'][name]['alpha'],
                    'scaling': manifest['adapter_info'][name]['scaling'],
                    'merged': manifest['adapter_info'][name]['merged'],
                }
                module.load_lora_state_dict(lora_state)
                loaded_modules.append(name)
            elif strict:
                raise KeyError(f"LoRA weights not found for module: {name}")
                
    print(f"LoRA Domain Adaptation: Loaded adapters into {len(loaded_modules)} modules")
    return manifest


def get_model_lora_info(model: nn.Module) -> Dict[str, Any]:
    """
    LoRA Domain Adaptation: Get information about LoRA adapters in the model.
    
    This function analyzes the model and provides detailed information about
    all LoRA adapters, including parameter counts, memory usage, and configuration.
    
    Args:
        model: The model to analyze
        
    Returns:
        Dict[str, Any]: Detailed information about LoRA adapters
    """
    lora_modules = {}
    total_lora_params = 0
    total_base_params = 0
    
    # LoRA Domain Adaptation: Analyze each module
    for name, module in model.named_modules():
        if isinstance(module, LoRAInjectedLinear):
            lora_params = module.lora_A.numel() + module.lora_B.numel()
            # 🔧 FIXED: Access base_linear.weight, not module.weight
            base_params = module.base_linear.weight.numel()
            if module.base_linear.bias is not None:
                base_params += module.base_linear.bias.numel()
                
            lora_modules[name] = {
                'rank': module.rank,
                'alpha': module.alpha,
                'scaling': module.scaling,
                'merged': module.merged,
                'lora_parameters': lora_params,
                'base_parameters': base_params,
                'compression_ratio': base_params / lora_params if lora_params > 0 else 0,
                'input_features': module.base_linear.in_features,
                'output_features': module.base_linear.out_features,
            }
            
            total_lora_params += lora_params
            total_base_params += base_params
    
    # LoRA Domain Adaptation: Calculate overall statistics
    info = {
        'total_lora_modules': len(lora_modules),
        'total_lora_parameters': total_lora_params,
        'total_base_parameters': total_base_params,
        'overall_compression_ratio': total_base_params / total_lora_params if total_lora_params > 0 else 0,
        'lora_percentage': (total_lora_params / (total_lora_params + total_base_params)) * 100 if (total_lora_params + total_base_params) > 0 else 0,
        'modules': lora_modules,
    }
    
    return info
