import os
import torch
import psutil 

def get_device():
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"

def is_gpu_available():
    return torch.cuda.is_available() or torch.backends.mps.is_available()


def get_scaler():
    if torch.cuda.is_available():
        return torch.cuda.amp.GradScaler()
    elif torch.backends.mps.is_available():
        return torch.amp.GradScaler("mps")
    else:
        return None

def get_gpu_memory_allocated_size():
    gpu_mem_mb = 0
    
    if torch.cuda.is_available():
        gpu_mem_mb = torch.cuda.memory_allocated()
    elif torch.backends.mps.is_available():
        process = psutil.Process(os.getpid())
        mem_bytes = process.memory_info().rss  # includes unified memory
        gpu_mem_mb = mem_bytes / (1024 * 1024)
    else:
        gpu_mem_mb = 0

    return gpu_mem_mb


# _GPT_CONFIG_OPAL_20M = {
#     "vocab_size": 12000,
#     "context_length": 1024,       # ↑ for longer prompts
#     "emb_dim": 320,               # ↑ better token representations
#     "n_heads": 8,                 # scales well with emb_dim
#     "n_layers": 12,               # ↑ more reasoning depth
#     "drop_rate": 0.05,            # lower dropout for accuracy
#     "transformer_drop_rate": 0.1,
#     "attention_drop_rate": 0.1,
#     "qkv_bias": False,
#     "num_epoch": 3,
#     "learning_rate": 3e-4,
#     "weight_decay": 0.1,
#     "early_stopping_patience": 2,
#     "persistent_workers": False,
#     "gradient_accumulation_steps": 1,  # ✅ Add explicitly
#     "max_grad_norm": 1.0,               # ✅ Add gradient clipping
#     "kv_heads" : 1,                # MQA
#     "use_rope": True,              # Rotary pos embeddings
#     "tie_embeddings": True          # Tie input/output embeddings
# }

_GPT_CONFIG_OPAL_45M = {
    "vocab_size": 12000,
    "context_length": 1024,       # ↑ for longer prompts
    "emb_dim": 512,               # ↑ better token representations
    "n_heads": 8,                 # scales well with emb_dim
    "n_layers": 12,               # ↑ more reasoning depth
    "drop_rate": 0.05,            # lower dropout for accuracy
    "transformer_drop_rate": 0.1,
    "attention_drop_rate": 0.1,
    "qkv_bias": True,
    "num_epoch": 5,
    "learning_rate": 2e-4,
    "weight_decay": 0.1,
    "early_stopping_patience": 2,
    "persistent_workers": False,
    "gradient_accumulation_steps": 4,  # ✅ Add explicitly
    "max_grad_norm": 1.0,               # ✅ Add gradient clipping
    "kv_heads" : 1,                # MQA
    "use_rope": True,              # Rotary pos embeddings
    "tie_embeddings": True,        # Tie input/output embeddings
    # ✅ FIX: Add special token IDs to prevent CUDA index out of bounds (matches sptrainer.py)
    "pad_id": 0,                   # Safe padding token
    "bos_id": 1,                   # Beginning of sequence token (matches tokenizer training)
    "eos_id": 2,                   # End of sequence token (matches tokenizer training)
    "unk_id": 3                    # Unknown token
}

# TRAINING RUNTIME (GPU)
_TRAINING_CONFIG_GPU = {
    "device": get_device(),
    "batch_size": 12,                 # smaller micro-batch for stability
    "num_workers": 2,
    "mixed_precision": True,          # AMP is fine if stable; set False if you see NaNs
    # If your trainer is steps-based, target 1–3k steps total for 10k samples, warmup 4%
}

GPT_CONFIG_OPAL_FINETUNE_45M = {
    "vocab_size": 12000,
    "context_length": 1024,
    "emb_dim": 512,
    "n_heads": 8,
    "n_layers": 12,
    # ↓ Slightly lighter regularization for SFT
    "drop_rate": 0.05,               # was 0.1
    "transformer_drop_rate": 0.10,   # was 0.15
    "attention_drop_rate": 0.05,     # was 0.10
    "qkv_bias": True,
    "num_epoch": 2,                  # OK (use steps-based stopping if possible)
    # ↓ LR & scheduling
    "learning_rate": 2e-7,           # was 2e-5
    "weight_decay": 0.005,
    "warmup_steps": None,            # use ratio in trainer (see below)
    "warmup_ratio": 0.04,            # 4% of total steps
    "early_stopping_patience": 3,    # was 1
    "persistent_workers": False,
    "gradient_accumulation_steps": 4, # aim for effective batch 32–128
    "lr_scheduler": "cosine",
    "max_grad_norm": 0.5,            # was 0.3
    "kv_heads": 1,
    "use_rope": True,
    "tie_embeddings": True,
    "pad_id": 0, "bos_id": 1, "eos_id": 2, "unk_id": 3,
    "commands_weight": 2.75,         # gentle bias to config/show blocks
    "label_smoothing": 0.02,         # add this if your trainer supports it
}



# 🍎 MPS-specific ultra-conservative configuration for Apple Silicon
_TRAINING_CONFIG_MPS = {
    'batch_size': 1,
    'gradient_accumulation_steps': 1,  # Reduced from 2 to 1 (no accumulation)
    'mixed_precision': False,
    'num_workers': 0,
    'learning_rate': 1e-5,
    'max_tokens_per_batch': 32,   # Reduced from 64 to 32 (extreme)
    'max_seq_length': 32,         # Reduced from 64 to 32 (extreme)
    'micro_batch_size': 1,        # Added micro-batching
    'use_gradient_checkpointing': True,  # Enable gradient checkpointing
}

# =====================================================
# 🚀 3-PHASE TRAINING CONFIGURATION SYSTEM
# =====================================================

# Phase-specific model configurations
_PHASE_CONFIGS = {
    "pretraining": {
        **_GPT_CONFIG_OPAL_45M,
        "learning_rate": 3e-4,        # Higher LR for initial pretraining
        "num_epoch": 2,               # 1-2 epochs sufficient for 5GB
        "early_stopping_patience": 3,
        "weight_decay": 0.1,
        "gradient_accumulation_steps": 4,
    },
    
    "domain_adaptation": {
        **_GPT_CONFIG_OPAL_45M,
        "learning_rate": 1e-4,        # Lower LR for domain adaptation
        "num_epoch": 3,               # More focused training on domain data
        "early_stopping_patience": 4,
        "weight_decay": 0.05,         # Reduced weight decay
        "gradient_accumulation_steps": 4,
    },
    
    "fine_tuning": {
        **GPT_CONFIG_OPAL_FINETUNE_45M,
        "learning_rate": 5e-5,        # Increased from 2e-7 for better convergence
        "num_epoch": 3,               # Sufficient for instruction following
        "early_stopping_patience": 5,
        "weight_decay": 0.01,         # Very low weight decay for fine-tuning
        "gradient_accumulation_steps": 2,  # Smaller accumulation for stability
    }
}

# Phase-specific training configurations
_TRAINING_CONFIGS = {
    "pretraining": {
        **_TRAINING_CONFIG_GPU,
        "batch_size": 16,             # Larger batches for pretraining efficiency
        "mixed_precision": True,      # Enable for speed on large corpus
        "num_workers": 2,
    },
    
    "domain_adaptation": {
        **_TRAINING_CONFIG_GPU,
        "batch_size": 12,             # Moderate batch size
        "mixed_precision": True,      # Keep enabled for efficiency
        "num_workers": 2,
    },
    
    "fine_tuning": {
        **_TRAINING_CONFIG_GPU,
        "batch_size": 8,              # Smaller batches for fine-tuning stability
        "mixed_precision": False,     # Disabled for stability in fine-tuning
        "num_workers": 0,             # No multiprocessing for fine-tuning
    }
}

# Current training phase (default to pretraining)
CURRENT_PHASE = "pretraining"

def set_training_phase(phase: str):
    """
    Set the current training phase and update configurations accordingly.
    
    Args:
        phase (str): One of ['pretraining', 'domain_adaptation', 'fine_tuning']
    """
    global OPAL_MODEL_CONFIG, TRAINING_CONFIG, CURRENT_PHASE
    
    valid_phases = ["pretraining", "domain_adaptation", "fine_tuning"]
    if phase not in valid_phases:
        raise ValueError(f"Invalid phase '{phase}'. Must be one of {valid_phases}")
    
    CURRENT_PHASE = phase
    OPAL_MODEL_CONFIG = _PHASE_CONFIGS[phase].copy()
    
    # Apply device-specific adjustments
    if torch.backends.mps.is_available():
        TRAINING_CONFIG = {
            **_TRAINING_CONFIG_MPS,
            "batch_size": 2 if phase == "pretraining" else 1,  # Slightly larger for pretraining
        }
        print(f"🍎 Using MPS-specific configuration for {phase}")
    else:
        TRAINING_CONFIG = _TRAINING_CONFIGS[phase].copy()
    
    # Print configuration summary
    phase_emoji = {"pretraining": "🚀", "domain_adaptation": "🎯", "fine_tuning": "🔧"}
    print(f"\n{phase_emoji[phase]} ===== SWITCHED TO {phase.upper().replace('_', ' ')} PHASE =====")
    print(f"📊 Model: {OPAL_MODEL_CONFIG['emb_dim']}D embedding, {OPAL_MODEL_CONFIG['n_layers']} layers")
    print(f"📊 Learning rate: {OPAL_MODEL_CONFIG['learning_rate']:.2e}")
    print(f"📊 Epochs: {OPAL_MODEL_CONFIG['num_epoch']}")
    print(f"📊 Batch size: {TRAINING_CONFIG['batch_size']}")
    print(f"📊 Gradient accumulation: {OPAL_MODEL_CONFIG['gradient_accumulation_steps']}")
    print(f"📊 Mixed precision: {TRAINING_CONFIG['mixed_precision']}")
    print(f"📊 Early stopping patience: {OPAL_MODEL_CONFIG['early_stopping_patience']}")
    print(f"{phase_emoji[phase]} ================================================\n")

def get_phase_description(phase: str) -> str:
    """Get description of what each training phase accomplishes."""
    descriptions = {
        "pretraining": "General language understanding from mixed corpus (5GB: 2GB Cisco + 3GB FineWeb-EDU)",
        "domain_adaptation": "Specialized Cisco domain knowledge from pure domain corpus (2GB Cisco docs)",
        "fine_tuning": "Task-specific instruction following with curated dataset"
    }
    return descriptions.get(phase, "Unknown phase")

# Initialize with pretraining configuration
set_training_phase("pretraining")


# _GPT_CONFIG_OPAL_FINETUNE_45M = {
#     **_GPT_CONFIG_OPAL_45M,
#     "drop_rate": 0.2,                 # 🔧 Increase dropout for more regularization
#     "transformer_drop_rate": 0.2,   
#     "attention_drop_rate": 0.15,

#     # 🔹 Fine-tuning Hyperparameters - Ultra conservative for overfitting prevention
#     "num_epoch": 3,                   # 🔧 Even fewer epochs 
#     "learning_rate": 1e-7,            # 🔧 Much much lower LR to slow learning
#     "weight_decay": 0.05,             # 🔧 More weight decay
#     "early_stopping_patience": 8,     # 🔧 More patience 
#     "gradient_accumulation_steps": 4, 
#     "max_grad_norm": 0.5,             # 🔧 Stricter gradient clipping
    
#     # 🔹 Fine-tuning specific settings
#     "warmup_steps": 500,              # 🔧 Much more warmup
#     "lr_scheduler": "cosine",
    
#     # ✅ FIX: Ensure special tokens are properly configured for fine-tuning
#     "pad_id": 0,                      # Safe padding token  
#     "bos_id": 1,                      # Beginning of sequence token (matches tokenizer training)
#     "eos_id": 2,                      # End of sequence token (matches tokenizer training)
#     "unk_id": 3                       # Unknown token
# }

# _GPT_CONFIG_OPAL_GPU_45M = {
#     **_GPT_CONFIG_OPAL_45M,
#     # You can override GPU-specific model parameters if needed
# }



# # -----------------------------
# # Define Training Configurations
# # -----------------------------

# _TRAINING_CONFIG_CPU = {
#     "device": "cpu",
#     "batch_size": 8,
#     "num_workers": 0,
#     "mixed_precision": False,
# }


# def set_finetune_mode(enable_finetune=True):
#     """
#     Helper function to switch between pretraining and fine-tuning configurations
    
#     Args:
#         enable_finetune (bool): If True, use fine-tuning configs; otherwise pretraining
#     """
#     global OPAL_MODEL_CONFIG, TRAINING_CONFIG, FINETUNE_MODE
    
#     FINETUNE_MODE = enable_finetune
#     USE_GPU = is_gpu_available()
    
#     if FINETUNE_MODE:
#         print("🔧 Switching to FINE-TUNING configuration...")
#         OPAL_MODEL_CONFIG = _GPT_CONFIG_OPAL_FINETUNE_45M
#         TRAINING_CONFIG = {
#             **(_TRAINING_CONFIG_GPU if USE_GPU else _TRAINING_CONFIG_CPU),
#             "batch_size": 4 if USE_GPU else 1,  # 🔧 Even smaller batches for stability
#             "mixed_precision": False,  # 🚨 DISABLED: Mixed precision can cause CUDA index errors
#             "num_workers": 0,  # 🚨 DISABLED: Prevent multiprocessing conflicts
#         }
#         print(f"   → Model: {OPAL_MODEL_CONFIG['emb_dim']}D, LR: {OPAL_MODEL_CONFIG['learning_rate']}")
#         print(f"   → Batch size: {TRAINING_CONFIG['batch_size']}, Epochs: {OPAL_MODEL_CONFIG['num_epoch']}")
#         print(f"   → Mixed precision: {TRAINING_CONFIG['mixed_precision']} (disabled for stability)")
#     else:
#         print("🚀 Switching to PRETRAINING configuration...")
#         OPAL_MODEL_CONFIG = _GPT_CONFIG_OPAL_45M
#         TRAINING_CONFIG = _TRAINING_CONFIG_GPU if USE_GPU else _TRAINING_CONFIG_CPU
#         print(f"   → Model: {OPAL_MODEL_CONFIG['emb_dim']}D, LR: {OPAL_MODEL_CONFIG['learning_rate']}")
#         print(f"   → Batch size: {TRAINING_CONFIG['batch_size']}, Epochs: {OPAL_MODEL_CONFIG['num_epoch']}")


# -----------------------------
# Select Environment
# -----------------------------

# USE_GPU = is_gpu_available()  # Change this if you want to force CPU/GPU

# # Configuration selection - set FINETUNE_MODE=True when fine-tuning
# FINETUNE_MODE = True  # ✅ Set this to True when fine-tuning

# if FINETUNE_MODE:
#     # Use fine-tuning optimized configs
#     OPAL_MODEL_CONFIG = _GPT_CONFIG_OPAL_FINETUNE_45M
#     # 🔧 FIXED: Conservative settings to prevent CUDA errors
#     TRAINING_CONFIG = {
#         **(_TRAINING_CONFIG_GPU if USE_GPU else _TRAINING_CONFIG_CPU),
#         "batch_size": 8 if USE_GPU else 2,  # 🔧 Smaller batches for stable fine-tuning
#         "mixed_precision": False,  # 🚨 DISABLED: Mixed precision can cause CUDA index errors
#         "num_workers": 0,  # 🚨 DISABLED: Prevent multiprocessing conflicts
#     }
# else:
#     # Use pretraining configs  
#     OPAL_MODEL_CONFIG = _GPT_CONFIG_OPAL_45M
#     TRAINING_CONFIG = _TRAINING_CONFIG_GPU if USE_GPU else _TRAINING_CONFIG_CPU


# # OPAL_MODEL_CONFIG = _GPT_CONFIG_OPAL_FINETUNE_45M
# # ---- Small-model efficiency flags (defaults) ----
# for _cfg_name, _cfg in list(globals().items()):
#     if isinstance(_cfg, dict) and _cfg.get("vocab_size") and _cfg.get("emb_dim"):
#         _cfg.setdefault("kv_heads", 1)         # MQA
#         _cfg.setdefault("use_rope", True)      # Rotary pos embeddings
#         _cfg.setdefault("tie_embeddings", True)
