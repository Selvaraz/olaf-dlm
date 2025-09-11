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
    "context_length": 512,       # ↑ for longer prompts
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
    "gradient_accumulation_steps": 1,  # ✅ Add explicitly
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

_GPT_CONFIG_OPAL_FINETUNE_45M = {
    "vocab_size": 12000,
    "context_length": 512,       # ↑ for longer prompts
    "emb_dim": 512,               # ↑ better token representations
    "n_heads": 8,                 # scales well with emb_dim
    "n_layers": 12,               # ↑ more reasoning depth
    "drop_rate": 0.1,             # 🔧 Reduced from 0.2 - less aggressive dropout
    "transformer_drop_rate": 0.15, # 🔧 Reduced from 0.2
    "attention_drop_rate": 0.1,   # 🔧 Reduced from 0.15
    "qkv_bias": True,
    "num_epoch": 2,               # 🔧 ULTRA-CONSERVATIVE: Reduced from 3 to prevent forgetting
    "learning_rate": 1e-7,        # 🔧 CRITICAL: Much lower to prevent catastrophic forgetting
    "weight_decay": 0.005,        # 🔧 Reduced from 0.01 - minimal regularization
    "warmup_steps": 100,          # 🔧 Adjusted for 2 epochs
    "early_stopping_patience": 1, # 🔧 Very quick stopping if overfitting
    "persistent_workers": False,
    "gradient_accumulation_steps": 8, # 🔧 Larger accumulation for stability
    "lr_scheduler": "cosine",
    "max_grad_norm": 0.3,         # 🔧 Much stricter gradient clipping
    "kv_heads" : 1,                # MQA
    "use_rope": True,              # Rotary pos embeddings
    "tie_embeddings": True,        # Tie input/output embeddings
    # ✅ FIX: Add special token IDs to prevent CUDA index out of bounds (matches sptrainer.py)
    "pad_id": 0,                   # Safe padding token
    "bos_id": 1,                   # Beginning of sequence token (matches tokenizer training)
    "eos_id": 2,                   # End of sequence token (matches tokenizer training)
    "unk_id": 3,                    # Unknown token
    "commands_weight": 3.0       # Weight boost for command tokens in fine-tuning
}



_TRAINING_CONFIG_GPU = {
    "device": get_device(),
    "batch_size": 4,              # 🔧 Even smaller batches for ultra-conservative training
    "num_workers": 2,             # 🔧 Reduced further for stability
    "mixed_precision": False,     # 🔧 Disabled for fine-tuning stability
    "gradient_accumulation_steps": 8  # 🔧 Matches model config for effective batch size 32
}

OPAL_MODEL_CONFIG = _GPT_CONFIG_OPAL_FINETUNE_45M
TRAINING_CONFIG = _TRAINING_CONFIG_GPU


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
