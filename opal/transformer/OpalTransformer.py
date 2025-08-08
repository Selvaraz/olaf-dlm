import torch.nn as nn

from .FeedForward import FeedForward
from .OpalLayerNormalization import OpalLayerNormalization
from ..attention.multihead_attention import MultiheadAttention

# Diagram for OpalTransformerBlock

# +---------------------------+
# |   OpalTransformerBlock    |
# |                           |
# |  +---------------------+  |
# |  |   Self-Attention    |  |
# |  |     (Multi-Head)    |  |
# |  +---------+-----------+  |
# |            |              |
# |            v              |
# |  +---------------------+  |
# |  |   Layer Norm        |  |
# |  +---------------------+  |
# |            |              |
# |            v              |
# |  +---------------------+  |
# |  |    Feed Forward     |  |
# |  +------+---+---+------+  |
# |         |   |   |         |
# |         v   v   v         |
# |      GELU  +   GELU       |
# |         |   |   |         |
# |         +---+---+         |
# |            |              |
# |            v              |
# |  +---------------------+  |
# |  |   Layer Norm        |  |
# |  +---------------------+  |
# |            |              |
class OpalTransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.mhAttention = MultiheadAttention(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            context_length=cfg["context_length"],
            dropout=cfg["attention_drop_rate"],
            num_heads=cfg["n_heads"],
            qkv_bias=cfg["qkv_bias"],
            kv_heads=cfg.get("kv_heads", 1),
            use_rope=cfg.get("use_rope", True)
        )
        self.feedForward = FeedForward(cfg)
        self.inputLayerNorm = OpalLayerNormalization(cfg["emb_dim"])
        self.outputLayerNorm = OpalLayerNormalization(cfg["emb_dim"])
        self.drop_shortcut = nn.Dropout(cfg["transformer_drop_rate"])

        
    def forward(self, x, past_key=None, past_value=None, use_cache=False):
        # cache: added (past_key, past_value, use_cache) so this block can thread KV-cache through attention
        #        - past_key/past_value: (B, H_exp, Tpast, Dh) or None
        #        - when use_cache=True, we’ll return present_key/present_value for the next decode step

        # Shortcut connection for residual learning
        shortcut = x
        # First layer normalization and multi-head attention
        x = self.inputLayerNorm(x)
        # Apply multi-head attention
        # cache: attention now returns (context, present_key, present_value) when use_cache=True,
        #        otherwise (context, None, None). We always unpack the triple.
        attn_out, present_key, present_value = self.mhAttention(
            x, past_key=past_key, past_value=past_value, use_cache=use_cache
        ) 
        # Apply dropout to the shortcut connection
        x = self.drop_shortcut(attn_out)
        # Add the shortcut connection
        x = x + shortcut

        # Second layer normalization and feed-forward network
        shortcut = x
        # Apply layer normalization
        x = self.outputLayerNorm(x)
        # Apply feed-forward network
        x = self.feedForward(x)
        # Apply dropout to the shortcut connection
        x = self.drop_shortcut(x)
        x = x + shortcut

        # cache: return present_key/present_value so the model can collect layer-wise caches
        return x, present_key, present_value
