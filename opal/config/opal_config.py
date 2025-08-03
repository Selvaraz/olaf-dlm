import torch

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

    # 🔹 Fine-tuning Hyperparameters
    "num_epoch": 8,                   # ✅ Fewer epochs to prevent overfitting
    "learning_rate": 3e-5,            # ✅ Slightly higher for better adaptation
    "weight_decay": 0.05,
    "early_stopping_patience": 3,
    "gradient_accumulation_steps": 2, # ✅ Useful if GPU memory is low
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
    "num_epoch": 20,
    "learning_rate": 3e-4,
    "weight_decay": 0.1,
    "early_stopping_patience": 6,
    "persistent_workers": False,
    "gradient_accumulation_steps": 1,  # ✅ Add explicitly
    "max_grad_norm": 1.0               # ✅ Add gradient clipping
}

_GPT_CONFIG_OPAL_FINETUNE_20M = {
    **_GPT_CONFIG_OPAL_20M,
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
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "batch_size": 64,
    "num_workers": 0,
    "mixed_precision": torch.cuda.is_available(),
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


# -----------------------------
# Select Environment
# -----------------------------

USE_GPU = torch.cuda.is_available()  # Change this if you want to force CPU/GPU

#OPAL_MODEL_CONFIG = _GPT_CONFIG_OPAL_GPU_8M if USE_GPU else _GPT_CONFIG_OPAL_CPU_8M
OPAL_MODEL_CONFIG = _GPT_CONFIG_OPAL_20M
TRAINING_CONFIG = _TRAINING_CONFIG_GPU if USE_GPU else _TRAINING_CONFIG_CPU
