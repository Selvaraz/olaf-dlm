
import torch

def build_rope_cache(seq_len, head_dim, base=10000.0, device="cpu", dtype=torch.float32):
    assert head_dim % 2 == 0, "head_dim must be even for RoPE"
    theta = base ** (-2 * torch.arange(0, head_dim//2, dtype=dtype, device=device) / head_dim)
    t = torch.arange(seq_len, dtype=dtype, device=device)[:, None]
    freqs = t * theta[None, :]
    return torch.cos(freqs), torch.sin(freqs)

def apply_rope(q, k, cos, sin):
    # q,k: (B,T,H,Dh); cos/sin: (T, Dh/2)
    Dh = q.shape[-1]
    q1, q2 = q[..., :Dh//2], q[..., Dh//2:]
    k1, k2 = k[..., :Dh//2], k[..., Dh//2:]
    # reshape cos/sin for broadcast: (1,T,1,Dh/2)
    cos = cos.unsqueeze(0).unsqueeze(2)
    sin = sin.unsqueeze(0).unsqueeze(2)
    q_rot = torch.cat([q1 * cos - q2 * sin, q1 * sin + q2 * cos], dim=-1)
    k_rot = torch.cat([k1 * cos - k2 * sin, k1 * sin + k2 * cos], dim=-1)
    return q_rot, k_rot
