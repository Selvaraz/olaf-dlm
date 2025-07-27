import torch
import torch.nn as nn

class MultiheadAttention(torch.nn.Module):
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
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        # Validate input parameters
        if (d_out%num_heads !=0 ):
            raise ValueError("d_out must be divisible by num_heads")
        
        # Initialize instance variables
        # d_in: Dimension of the input embeddings
        # d_out: Dimension of the output embeddings
        # context_length: Length of the context for the attention mechanism
        # dropout: Dropout rate to prevent overfitting
        # num_heads: Number of attention heads
        # qkv_bias: Whether to use bias in the query, key, and value matrices
        self.num_heads = num_heads
        self.d_out = d_out
        
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
        self.Wk = torch.nn.Linear(d_in, d_out, bias=qkv_bias)
        self.Wv = torch.nn.Linear(d_in, d_out, bias=qkv_bias)

        self.out_proj = nn.Linear(d_out, d_out)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer(
            "mask",
            torch.triu(torch.ones(context_length, context_length),
            diagonal=1)
        )

    def forward(self, x):
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
        keys = self.Wk(x)
        queries = self.Wq(x)
        values = self.Wv(x)

        # Reshape the keys, queries, and values to have the shape
        # (batch, num_tokens, num_heads, head_dim)
        # This is done to split the output embeddings into multiple attention heads.
        # The keys, queries, and values are reshaped to have the shape
        # (batch, num_tokens, num_heads, head_dim) so that they can be
        # used to compute the attention scores.
        # The keys, queries, and values are then transposed to have the shape
        # (batch, num_heads, num_tokens, head_dim) so that they can be
        # used to compute the attention scores.
        # Here we are splitting the matrices into multiple heads by adding an extra dimension num_heads.
        # Unroll the (batch, num_tokens, d_out) tensor into (batch, num_tokens, num_heads, head_dim).
        keys = keys.view(batch, num_tokens, self.num_heads, self.head_dim)
        values = values.view(batch, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(batch, num_tokens, self.num_heads, self.head_dim)

        # Transpose the keys, queries, and values to have the shape
        # from (batch, num_tokens, num_heads, head_dim) to (batch, num_heads, num_tokens, head_dim) 
        # This is done to split the output embeddings into multiple attention heads.
        # The keys, queries, and values are then transposed to have the shape
        # (batch, num_heads, num_tokens, head_dim) so that they can be
        # used to compute the attention scores. This is to group the tokens by heads.
        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)

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
        return context_vec