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
    "num_epoch": 20,
    "learning_rate": 3e-4,
    "weight_decay": 0.2,
    "early_stopping_patience": 6,
    "persistent_workers" : False
}

_GPT_CONFIG_OPAL_FINETUNE = {
    "vocab_size": 8000,              # Must match pretrained model
    "context_length": 768,           # Same as pretrained
    "emb_dim": 200,                  # Same as pretrained
    "n_heads": 5,                    # Same as pretrained
    "n_layers": 10,                  # Same as pretrained
    "drop_rate": 0.05,               # Slightly lower dropout for fine-tuning
    "transformer_drop_rate": 0.05,   # Reduce dropout to retain learned weights
    "attention_drop_rate": 0.05,     # Reduce dropout
    "qkv_bias": False,               # Must match pretrained

    # 🔹 Fine-tuning Hyperparameters
    "num_epoch": 10,                 # Fewer epochs to prevent overfitting
    "learning_rate": 5e-5,           # Lower LR for stable fine-tuning
    "weight_decay": 0.05,            # Smaller weight decay
    "early_stopping_patience": 3,    # Stop earlier if no improvement
    "persistent_workers": False,     # Keep as False for CPU training

    # Optional for better fine-tuning
    "warmup_steps": 200,             # Gradual LR warmup
    "gradient_accumulation_steps": 2,# If batch size is small
    "max_grad_norm": 1.0             # Gradient clipping for stability
}


_GPT_CONFIG_OPAL_GPU = {
    **_GPT_CONFIG_OPAL_CPU,
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

OPAL_MODEL_CONFIG = _GPT_CONFIG_OPAL_GPU if USE_GPU else _GPT_CONFIG_OPAL_CPU
TRAINING_CONFIG = _TRAINING_CONFIG_GPU if USE_GPU else _TRAINING_CONFIG_CPU
