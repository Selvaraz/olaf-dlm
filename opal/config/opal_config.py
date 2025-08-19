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

# -----------------------------
# Define Model Configurations
# -----------------------------
_GPT_CONFIG_OPAL_CPU_8M = {
    "vocab_size": 8000,
    "context_length": 768,
    "emb_dim": 200,
    "n_heads": 5,
    "n_layers": 10,
    "drop_rate": 0.1,
    "transformer_drop_rate": 0.1,
    "attention_drop_rate": 0.1,
    "qkv_bias": False,
    "num_epoch": 20,
    "learning_rate": 3e-4,
    "weight_decay": 0.1,               # ✅ Slightly lower
    "early_stopping_patience": 6,
    "persistent_workers": False,
    "gradient_accumulation_steps": 1,  # ✅ Add explicitly
    "max_grad_norm": 1.0               # ✅ Add gradient clipping
}

_GPT_CONFIG_OPAL_FINETUNE_8M = {
    **_GPT_CONFIG_OPAL_CPU_8M,
    "drop_rate": 0.05,
    "transformer_drop_rate": 0.05,
    "attention_drop_rate": 0.05,

    # 🔹 Fine-tuning Hyperparameters - Optimized for smaller model
    "num_epoch": 6,                   # ✅ Slightly more epochs for smaller model
    "learning_rate": 3e-5,            # ✅ Can handle slightly higher LR
    "weight_decay": 0.01,             # ✅ Reduced weight decay
    "early_stopping_patience": 3,
    "gradient_accumulation_steps": 2, 
    "max_grad_norm": 0.5,             # ✅ Add gradient clipping
    
    # 🔹 Fine-tuning specific settings
    "warmup_steps": 50,               # ✅ Less warmup for smaller model
    "lr_scheduler": "cosine",
}

_GPT_CONFIG_OPAL_20M = {
    "vocab_size": 8000,
    "context_length": 1024,       # ↑ for longer prompts
    "emb_dim": 320,               # ↑ better token representations
    "n_heads": 8,                 # scales well with emb_dim
    "n_layers": 12,               # ↑ more reasoning depth
    "drop_rate": 0.1,            # lower dropout for accuracy
    "transformer_drop_rate": 0.1,
    "attention_drop_rate": 0.1,
    "qkv_bias": False,
    "num_epoch": 3,
    "learning_rate": 8e-4,
    "weight_decay": 0.1,
    "early_stopping_patience": 2,
    "persistent_workers": False,
    "gradient_accumulation_steps": 1,  # ✅ Add explicitly
    "max_grad_norm": 1.0,               # ✅ Add gradient clipping
    "kv_heads" : 1,                # MQA
    "use_rope": True,              # Rotary pos embeddings
    "tie_embeddings": True          # Tie input/output embeddings
}

_GPT_CONFIG_OPAL_FINETUNE_20M = {
    **_GPT_CONFIG_OPAL_20M,
    "drop_rate": 0.05,               # ✅ Lower dropout for fine-tuning
    "transformer_drop_rate": 0.05,   
    "attention_drop_rate": 0.05,

    # 🔹 Fine-tuning Hyperparameters - Optimized for 100K dataset
    "num_epoch": 5,                   # ✅ Reduced to prevent overfitting on large dataset
    "learning_rate": 2e-5,            # ✅ Lower LR for stable fine-tuning
    "weight_decay": 0.01,             # ✅ Reduced weight decay
    "early_stopping_patience": 2,     # ✅ Earlier stopping for large dataset
    "gradient_accumulation_steps": 4, # ✅ Higher accumulation for effective larger batch
    "max_grad_norm": 0.5,             # ✅ Tighter gradient clipping for stability
    
    # 🔹 Fine-tuning specific settings
    "warmup_steps": 100,              # ✅ Warmup for stable training start
    "lr_scheduler": "cosine",         # ✅ Cosine annealing for fine-tuning
}


_GPT_CONFIG_OPAL_92M = {
    "vocab_size": 8000,
    "context_length": 1280,       # ↑ for longer prompts
    "emb_dim": 768,               # ↑ better token representations
    "n_heads": 12,                 # scales well with emb_dim
    "n_layers": 12,               # ↑ more reasoning depth
    "drop_rate": 0.1,            # lower dropout for accuracy
    "transformer_drop_rate": 0.1,
    "attention_drop_rate": 0.1,
    "qkv_bias": False,
    "num_epoch": 20,
    "learning_rate": 3e-4,
    "weight_decay": 0.1,
    "early_stopping_patience": 10,
    "persistent_workers": False,
    "gradient_accumulation_steps": 1,  # ✅ Add explicitly
    "max_grad_norm": 1.0               # ✅ Add gradient clipping
}

_GPT_CONFIG_OPAL_FINETUNE_92M = {
    **_GPT_CONFIG_OPAL_92M,
    "drop_rate": 0.05,
    "transformer_drop_rate": 0.05,
    "attention_drop_rate": 0.05,

    # 🔹 Fine-tuning Hyperparameters
    "num_epoch": 8,                   # ✅ Fewer epochs to prevent overfitting
    "learning_rate": 3e-5,            # ✅ Slightly higher for better adaptation
    "weight_decay": 0.05,
    "early_stopping_patience": 3,
    "gradient_accumulation_steps": 2, # ✅ Useful if GPU memory is low
}


_GPT_CONFIG_OPAL_GPU_8M = {
    **_GPT_CONFIG_OPAL_CPU_8M,
    # You can override GPU-specific model parameters if needed
}

_GPT_CONFIG_OPAL_GPU_20M = {
    **_GPT_CONFIG_OPAL_20M,
    # You can override GPU-specific model parameters if needed
}


_TRAINING_CONFIG_GPU = {
    "device": get_device(),
    "batch_size": 64,
    "num_workers": 0,
    "mixed_precision": is_gpu_available(),
    "gradient_accumulation_steps": 1
}


# -----------------------------
# Define Training Configurations
# -----------------------------

_TRAINING_CONFIG_CPU = {
    "device": "cpu",
    "batch_size": 8,
    "num_workers": 0,
    "mixed_precision": False,
    "gradient_accumulation_steps": 4
}


def set_finetune_mode(enable_finetune=True):
    """
    Helper function to switch between pretraining and fine-tuning configurations
    
    Args:
        enable_finetune (bool): If True, use fine-tuning configs; otherwise pretraining
    """
    global OPAL_MODEL_CONFIG, TRAINING_CONFIG, FINETUNE_MODE
    
    FINETUNE_MODE = enable_finetune
    USE_GPU = is_gpu_available()
    
    if FINETUNE_MODE:
        print("🔧 Switching to FINE-TUNING configuration...")
        OPAL_MODEL_CONFIG = _GPT_CONFIG_OPAL_FINETUNE_20M
        TRAINING_CONFIG = {
            **(_TRAINING_CONFIG_GPU if USE_GPU else _TRAINING_CONFIG_CPU),
            "batch_size": 16 if USE_GPU else 4,
        }
        print(f"   → Model: {OPAL_MODEL_CONFIG['emb_dim']}D, LR: {OPAL_MODEL_CONFIG['learning_rate']}")
        print(f"   → Batch size: {TRAINING_CONFIG['batch_size']}, Epochs: {OPAL_MODEL_CONFIG['num_epoch']}")
    else:
        print("🚀 Switching to PRETRAINING configuration...")
        OPAL_MODEL_CONFIG = _GPT_CONFIG_OPAL_20M
        TRAINING_CONFIG = _TRAINING_CONFIG_GPU if USE_GPU else _TRAINING_CONFIG_CPU
        print(f"   → Model: {OPAL_MODEL_CONFIG['emb_dim']}D, LR: {OPAL_MODEL_CONFIG['learning_rate']}")
        print(f"   → Batch size: {TRAINING_CONFIG['batch_size']}, Epochs: {OPAL_MODEL_CONFIG['num_epoch']}")


# -----------------------------
# Select Environment
# -----------------------------

USE_GPU = is_gpu_available()  # Change this if you want to force CPU/GPU

# Configuration selection - set FINETUNE_MODE=True when fine-tuning
FINETUNE_MODE = False  # ✅ Set this to True when fine-tuning

if FINETUNE_MODE:
    # Use fine-tuning optimized configs
    OPAL_MODEL_CONFIG = _GPT_CONFIG_OPAL_FINETUNE_20M
    # Reduce batch size for fine-tuning to prevent overfitting
    TRAINING_CONFIG = {
        **(_TRAINING_CONFIG_GPU if USE_GPU else _TRAINING_CONFIG_CPU),
        "batch_size": 16 if USE_GPU else 4,  # Smaller batches for fine-tuning
    }
else:
    # Use pretraining configs
    OPAL_MODEL_CONFIG = _GPT_CONFIG_OPAL_20M
    TRAINING_CONFIG = _TRAINING_CONFIG_GPU if USE_GPU else _TRAINING_CONFIG_CPU


# ---- Small-model efficiency flags (defaults) ----
for _cfg_name, _cfg in list(globals().items()):
    if isinstance(_cfg, dict) and _cfg.get("vocab_size") and _cfg.get("emb_dim"):
        _cfg.setdefault("kv_heads", 1)         # MQA
        _cfg.setdefault("use_rope", True)      # Rotary pos embeddings
        _cfg.setdefault("tie_embeddings", True)
