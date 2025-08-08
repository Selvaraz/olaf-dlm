import torch
import torch.nn as nn
from .rope_utils import build_rope_cache, apply_rope

class MultiheadAttention(nn.Module):
    """
    This class implements the Multihead Attention layer as described in 
    the paper "Attention is All You Need" by Vaswani et al. 2017.
    The Multihead Attention layer is an extension of the Self-Attention layer, 
    it allows the model to attend to multiple representations of the same sequence 
    simultaneously and weigh their importance.

    The Multihead Attention layer takes in a sequence of input vectors and outputs a 
    sequence of output vectors.

    The Multihead Attention layer is defined by the following hyperparameters:
        - d_in: Dimension of the input embeddings
        - d_out: Dimension of the output embeddings
        - context_length: Length of the context for the attention mechanism
        - dropout: Dropout rate to prevent overfitting
        - num_heads: Number of attention heads
        - qkv_bias: Whether to use bias in the query, key, and value matrices

    Example usage:
        # Create a Multihead Attention layer with 4 attention heads and a context length of 256
        mha = MultiheadAttention(d_in=128, d_out=128, context_length=256, num_heads=4)

        # Create a sequence of input vectors
        inputs = torch.randn(1, 256, 128)

        # Pass the input sequence through the Multihead Attention layer
        outputs = mha(inputs)
    """
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False,
                 kv_heads=1, use_rope=True, rope_theta=10000.0, enable_new_attention=True):
        super().__init__()
        assert d_in == d_out, "d_in must equal d_out"
        # Validate input parameters
        if (d_out % num_heads != 0):
            raise ValueError("d_out must be divisible by num_heads")
        if (num_heads % max(1, kv_heads) != 0):
            raise ValueError("num_heads must be divisible by kv_heads")

        # Initialize instance variables
        # d_in: Dimension of the input embeddings
        # d_out: Dimension of the output embeddings
        # context_length: Length of the context for the attention mechanism
        # dropout: Dropout rate to prevent overfitting
        # num_heads: Number of attention heads
        # qkv_bias: Whether to use bias in the query, key, and value matrices
        self.num_heads = num_heads
        self.kv_heads = kv_heads        # rope: number of shared K/V heads (1 => MQA; >1 => GQA)
        self.d_out = d_out
        self.use_rope = use_rope        # rope: toggle rotary positional embeddings
        self.rope_theta = rope_theta    # rope: base frequency for RoPE
        self.context_length = context_length
        self.enable_new_attention = enable_new_attention  # rope: master switch to enable/disable new path

        # Calculate the dimension of each attention head
        # This is the dimension of the output embeddings divided by the number of attention heads.
        # This will be used to split the output embeddings into multiple attention heads.
        # For example, if d_out is 64 and num_heads is 8, then
        # head_dim will be 64/8 = 8.
        # This means that each attention head will have a dimension of 8.
        self.head_dim = d_out // num_heads

        # Calculate the scale factor for the attention scores
        # The scale factor is used to normalize the attention scores.
        # The scale factor is calculated as the square root of the head dimension.
        self.scale = self.head_dim ** -0.5 

        # Create linear layers for the query, key, and value matrices
        # These layers will transform the input embeddings into the query, key, and value matrices.
        # The query, key, and value matrices are used to compute the attention scores.
        # The query matrix is used to compute the attention scores for each token in the input sequence
        # The key matrix is used to compute the attention scores for each token in the input sequence
        # The value matrix is used to compute the final output of the attention mechanism

        self.Wq = torch.nn.Linear(d_in, d_out, bias=qkv_bias)
        # rope: When new attention is enabled, K/V heads can be shared via kv_heads.
        # rope: When disabled, we fall back to producing full K/V with num_heads (old behavior).
        kv_out = (self.kv_heads * self.head_dim) if enable_new_attention else d_out
        self.Wk = torch.nn.Linear(d_in, kv_out, bias=qkv_bias)
        self.Wv = torch.nn.Linear(d_in, kv_out, bias=qkv_bias)

        self.out_proj = nn.Linear(d_out, d_out)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer(
            "mask",
            torch.triu(torch.ones(context_length, context_length),
            diagonal=1)
        )

        # rope: lazily-built RoPE caches (cos/sin) for max context length
        self.register_buffer("rope_cos", None, persistent=False)
        self.register_buffer("rope_sin", None, persistent=False)

    def build_rope(self, T: int, device, dtype):
        """rope: Build (or extend) RoPE caches up to length T."""
        if not (self.enable_new_attention and self.use_rope):
            return
        need = (self.rope_cos is None) or (self.rope_cos.shape[0] < T)
        if need:
            cos, sin = build_rope_cache(max(T, self.context_length), self.head_dim, base=self.rope_theta, device=device, dtype=dtype)
            self.rope_cos = cos   # (T_max, head_dim/2)
            self.rope_sin = sin   # (T_max, head_dim/2)

    def forward(self, x, past_key=None, past_value=None, use_cache=False, **kwargs):
        """
        Forward with optional KV-cache.
        - past_key/past_value: (B, H_exp, Tpast, Dh) or None
        - use_cache=True: returns (context, present_key, present_value)
        - use_cache=False: returns (context, None, None)

        Backward-compat aliases:
        - accepts kwargs 'past_k', 'past_v' (old names) and 'mask' (legacy)
        """
        # cache: accept legacy argument names to prevent call-site errors
        if past_key is None and "past_k" in kwargs:
            past_key = kwargs["past_k"]
        if past_value is None and "past_v" in kwargs:
            past_value = kwargs["past_v"]
        _legacy_mask = kwargs.get("mask", None)  # unused; we derive mask from sizes

        batch, num_tokens, d_in = x.shape

        # Calculate the keys, queries, and values
        # The keys, queries, and values are computed by passing the input embeddings
        # through the linear layers defined in the constructor.
        # The keys, queries, and values are used to compute the attention scores.
        # The keys, queries, and values are computed by passing the input embeddings
        # through the linear layers defined in the constructor.
        # The keys, queries, and values are then reshaped to have the shape
        # (batch, num_tokens, d_out) so that they can be used to compute
        # the attention scores.
        queries = self.Wq(x)
        keys = self.Wk(x)
        values = self.Wv(x)

        if not self.enable_new_attention:
            # ===== Old code path (preserves original behavior; no RoPE, full K/V heads) =====
            # Reshape to (batch, num_tokens, num_heads, head_dim) using full d_out for K/V as well.
            keys = keys.view(batch, num_tokens, self.num_heads, self.head_dim)
            values = values.view(batch, num_tokens, self.num_heads, self.head_dim)
            queries = queries.view(batch, num_tokens, self.num_heads, self.head_dim)

            # Transpose to (batch, num_heads, num_tokens, head_dim) to compute attention scores
            keys = keys.transpose(1, 2)
            queries = queries.transpose(1, 2)
            values = values.transpose(1, 2)

            attn_scores = queries @ keys.transpose(2, 3)

            mask_bool = self.mask.bool()[:num_tokens, :num_tokens]
            attn_scores.masked_fill_(mask_bool, -torch.inf)

            attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
            attn_weights = self.dropout(attn_weights)

            context_vec = (attn_weights @ values).transpose(1, 2)
            context_vec = context_vec.contiguous().view(batch, num_tokens, self.d_out)

            context_vec = self.out_proj(context_vec)
            return context_vec

        # ===== New code path: RoPE + optional MQA/GQA =====
        # Reshape queries to (batch, num_tokens, num_heads, head_dim)
        queries = queries.view(batch, num_tokens, self.num_heads, self.head_dim)
        # rope: keys/values may have shared kv_heads when kv_heads < num_heads
        keys = keys.view(batch, num_tokens, self.kv_heads, self.head_dim)
        values = values.view(batch, num_tokens, self.kv_heads, self.head_dim)

        # rope: apply rotary embeddings to Q and K before any transposes (expects (B,T,H,Dh))
        if self.use_rope:
            self.build_rope(num_tokens, x.device, x.dtype)
            cos = self.rope_cos[:num_tokens]  # (T, Dh/2)
            sin = self.rope_sin[:num_tokens]  # (T, Dh/2)
            queries, keys = apply_rope(queries, keys, cos, sin)

        # rope: expand shared K/V across query heads if kv_heads < num_heads (MQA/GQA)
        if self.kv_heads == 1:
            # rope: broadcast K/V to all heads without allocating (view-based expand)
            keys = keys.expand(batch, num_tokens, self.num_heads, self.head_dim)
            values = values.expand(batch, num_tokens, self.num_heads, self.head_dim)
        else:
            # rope: repeat each kv head across groups of query heads (GQA)
            reps = self.num_heads // self.kv_heads
            keys = keys.repeat_interleave(reps, dim=2)
            values = values.repeat_interleave(reps, dim=2)

        # Transpose the keys, queries, and values to have the shape
        # from (batch, num_tokens, num_heads, head_dim) to (batch, num_heads, num_tokens, head_dim) 
        # This is done to split the output embeddings into multiple attention heads.
        # The keys, queries, and values are then transposed to have the shape
        # (batch, num_heads, num_tokens, head_dim) so that they can be
        # used to compute the attention scores. This is to group the tokens by heads.
        queries = queries.transpose(1, 2)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)

        # cache: Append past K/V (concatenate over time dim)
        if (past_key is not None) and (past_value is not None):
            # Ensure dtype/device match (if caches are fp16 and model is fp32/int8)
            keys   = torch.cat([past_key.to(keys.dtype),   keys],   dim=2)  # (B,H,Tpast+T,Dh)
            values = torch.cat([past_value.to(values.dtype), values], dim=2)

        # Compute the scaled dot-product attention scores,
        # The attention scores are computed by taking the dot product of the queries and keys.
        # The attention scores are then scaled by the square root of the head dimension.
        # The reason for transposing the keys with 2,3 is to align the dimensions for the dot product
        # the 2 represents the num_tokens and the 3 represents the head_dim.
        # The resultant dimention of attn_scores will be (batch, num_heads, num_tokens, num_tokens)
        # The dimention for queries is (batch, num_heads, num_tokens, head_dim)
        # The dimention for transposed (2,3) keys is (batch, num_heads, head_dim, num_tokens). So this is
        # the dot product for each head.
        attn_scores = queries @ keys.transpose(2, 3)

        # Apply the mask to the attention scores
        # The mask is applied to the attention scores to prevent the model from attending to future tokens
        # The mask is a triangular matrix that has 1s in the lower triangle and 0s in the upper triangle.
        # The mask is applied to the attention scores to prevent the model from attending to future tokens.
        # The mask is applied to the attention scores by setting the attention scores to -inf for 
        # the positions that are masked.
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]
        attn_scores.masked_fill_(mask_bool, -torch.inf)

        # Compute the softmax of the attention scores, this is to normalize the attention scores
        # The softmax is applied to the attention scores to normalize them. 
        # The intuition behind keys.shape[-1]**0.5 is to scale the attention scores 
        # by the square root of the head dimension.
        # This is done to prevent the attention scores from becoming too large or too small.
        # The dim=-1 is to apply the scaling to the last dimension of the attention scores.
        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)

        # Apply dropout to the attention weights
        # The dropout is applied to the attention weights to prevent overfitting.
        attn_weights = self.dropout(attn_weights)

        # Compute the context vector by multiplying the attention weights with the values
        # The context vector is computed by multiplying the attention weights with the values.
        # The context vector is the final output of the attention mechanism. The intution behing
        # this is to combine the information from the values based on the attention weights. The
        # Intuition behind transpose(1, 2) is to align the dimensions for the dot product.
        # The resultant dimention of context_vec will be (batch, num_heads, num_tokens, head_dim)
        context_vec = (attn_weights @ values).transpose(1, 2)
        # Reshape the context vector to have the shape (batch, num_tokens, d_out)
        # This is done to combine the information from the multiple attention heads into a single output vector
        context_vec = context_vec.contiguous().view(batch, num_tokens, self.d_out)

        # Apply the output projection to the context vector
        # The output projection is a linear layer that transforms the context vector into the final output vector
        context_vec = self.out_proj(context_vec)
        
        if use_cache:
            # cache: return present caches in fp16 to reduce RAM
            present_key   = keys.to(torch.float16)
            present_value = values.to(torch.float16)
            return context_vec, present_key, present_value
        else:
            return context_vec, None, None