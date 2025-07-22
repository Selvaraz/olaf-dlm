import torch
from ..attention import self_attention
from ..attention import causal_attention
from ..attention import multihead_attention

torch.manual_seed(1234)  # For reproducibility
# Create a random input tensor of shape (batch_size, sequence_length, d_in)
#inputs = torch.rand(batch_size, sequence_length, d_in)
inputs = torch.tensor(
    [[0.43, 0.15, 0.89], # Your (x^1)
    [0.55, 0.87, 0.66], # journey (x^2)
    [0.57, 0.85, 0.64], # starts (x^3)
    [0.22, 0.58, 0.33], # with (x^4)
    [0.77, 0.25, 0.10], # one (x^5)
    [0.05, 0.80, 0.55]] # step (x^6)
)

def test_multihead_attention():
    torch.manual_seed(123)
    batch = torch.stack((inputs, inputs), dim=0)
    batch_size, context_length, d_in = batch.shape
    d_out = 2
    mha = multihead_attention.MultiheadAttention(
            d_in, d_out, context_length, 0.0, num_heads=2, qkv_bias=True)
    context_vecs = mha(batch)
    print(context_vecs)
    print("context_vecs.shape:", context_vecs.shape)

def test_causal_attention():
    d_in = 3
    d_out = 3
    batch = torch.stack((inputs, inputs), dim=0)
    context_length = batch.shape[1]
    print("batch shape:", batch.shape)
    context_length = batch.shape[1]
    ca = causal_attention.CausalAttention(d_in, d_out, context_length, 0.0)
    context_vecs = ca(batch)
    print("context_vecs.shape:", context_vecs.shape)
    print(context_vecs)

def test_self_attention():
    d_in = 3
    d_out = 3
    batch_size = 1
    sequence_length = 6
    # Initialize the self-attention model
    model = self_attention.SelfAttentionV1(d_in, d_out)
    print("Attention output:")
    print(model(inputs))
    

# test_self_attention()
# test_causal_attention()
test_multihead_attention()