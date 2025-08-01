import torch

# -----------------------------
# Define Model Configurations
# -----------------------------

_GPT_CONFIG_OPAL_CPU = {
    "vocab_size": 8000,
    "context_length": 768,
    "emb_dim": 200,
    "n_heads": 5,
    "n_layers": 10,
    "drop_rate": 0.1,
    "transformer_drop_rate": 0.1,
    "attention_drop_rate": 0.1,
    "qkv_bias": False,
    "num_epoch": 50,
    "learning_rate": 4e-4,
    "weight_decay": 0.2,
    "early_stopping_patience": 10
}

_GPT_CONFIG_OPAL_GPU = {
    **_GPT_CONFIG_OPAL_CPU,
    # You can override GPU-specific model parameters if needed
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

_TRAINING_CONFIG_GPU = {
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "batch_size": 64,
    "num_workers": 4,
    "mixed_precision": torch.cuda.is_available(),
    "gradient_accumulation_steps": 1
}

# -----------------------------
# Select Environment
# -----------------------------

USE_GPU = torch.cuda.is_available()  # Change this if you want to force CPU/GPU

OPAL_MODEL_CONFIG = _GPT_CONFIG_OPAL_GPU if USE_GPU else _GPT_CONFIG_OPAL_CPU
TRAINING_CONFIG = _TRAINING_CONFIG_GPU if USE_GPU else _TRAINING_CONFIG_CPU
