def estimate_training_time_from_config(config, total_tokens, batch_size, hardware="cpu_36cores", epochs=10):
    """
    Estimate training time for a GPT-like model using a config dictionary.
    """

    vocab_size = config["vocab_size"]
    context_length = config["context_length"]
    emb_dim = config["emb_dim"]
    n_layers = config["n_layers"]

    # Parameter estimation
    emb_params = vocab_size * emb_dim
    lm_head_params = emb_dim * vocab_size
    attn_params_per_layer = 4 * emb_dim * emb_dim  # QKV + output projection
    ff_params_per_layer = 4 * emb_dim * emb_dim
    per_layer_params = attn_params_per_layer + ff_params_per_layer
    total_params = emb_params + lm_head_params + n_layers * per_layer_params

    # Steps per epoch
    tokens_per_batch = batch_size * context_length
    steps_per_epoch = total_tokens // tokens_per_batch

    # Time per batch heuristic
    time_per_batch_cpu = 0.3  # seconds
    time_per_batch_gpu = 0.03  # seconds
    time_per_batch = time_per_batch_cpu if "cpu" in hardware else time_per_batch_gpu

    epoch_time = steps_per_epoch * time_per_batch
    total_time = epoch_time * epochs

    return {
        "total_params_million": round(total_params / 1e6, 3),
        "steps_per_epoch": steps_per_epoch,
        "epoch_time_minutes": round(epoch_time / 60, 2),
        "total_time_hours": round(total_time / 3600, 2)
    }
