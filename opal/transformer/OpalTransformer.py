import torch.nn as nn

from .FeedForward import FeedForward
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
            qkv_bias=cfg["qkv_bias"]
        )
        self.feedForward = FeedForward(cfg)
        self.inputLayerNorm = nn.LayerNorm(cfg["emb_dim"])
        self.outputLayerNorm = nn.LayerNorm(cfg["emb_dim"])
        self.drop_shortcut = nn.Dropout(cfg["transformer_drop_rate"])

        
    def forward(self, x):

        # Shortcut connection for residual learning
        shortcut = x
        # First layer normalization and multi-head attention
        x = self.inputLayerNorm(x)
        # Apply multi-head attention
        x = self.mhAttention(x) 
        # Apply dropout to the shortcut connection
        x = self.drop_shortcut(x)
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

        return x