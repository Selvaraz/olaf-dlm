
import torch.nn as nn
import torch.nn.functional as F

class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        d = cfg["emb_dim"]
        inner = int(4 * d)
        self.w1 = nn.Linear(d, inner, bias=False)
        self.w2 = nn.Linear(d, inner, bias=False)  # gate
        self.w3 = nn.Linear(inner, d, bias=False)
    def forward(self, x):
        return self.w3(F.silu(self.w1(x)) * self.w2(x))
