
# Model config
_GPT_CONFIG_OPAL_FINAL = {
    "vocab_size": 8000,       # Enough for Cisco CLI + logs
    "context_length": 768,    # Can handle moderate logs
    "emb_dim": 192,           # Rich token embeddings
    "n_heads": 6,             # Balanced attention diversity
    "n_layers": 10,           # Enough depth for summarization
    "drop_rate": 0.1,
    "transformer_drop_rate": 0.1,
    "attention_drop_rate": 0.1,
    "qkv_bias": False
}

_GPT_CONFIG_OPAL_TEMP = {
    "vocab_size": 8000,       # Enough for Cisco CLI + logs
    "context_length": 768,    # Can handle moderate logs
    "emb_dim": 200,           # Rich token embeddings
    "n_heads": 5,             # Balanced attention diversity
    "n_layers": 10,           # Enough depth for summarization
    "drop_rate": 0.1,
    "transformer_drop_rate": 0.1,
    "attention_drop_rate": 0.1,
    "qkv_bias": False,
    "num_epoch": 500,
    "learning_rate": 4e-4,
    "weight_decay": 0.2
}

OPAL_MODEL_CONFIG = _GPT_CONFIG_OPAL_TEMP
