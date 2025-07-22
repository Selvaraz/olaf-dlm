import torch
import torch.nn as nn

class SelfAttentionV1(nn.Module):
    def __init__(self, d_in, d_out,  context_len=128, qkv_bias=False):
        super(SelfAttentionV1, self).__init__()

        # Initialize the input and output dimensions, context length, dropout rate,
        # and whether to use bias in the query, key, and value matrices.
        # d_in: Dimension of the input embeddings
        # d_out: Dimension of the output embeddings
        # context_len: Length of the context for the attention mechanism
        # dropout_percent: Dropout rate to prevent overfitting
        # qkv_bias: Whether to use bias in the query, key, and value matrices
        self.d_in = d_in
        self.d_out = d_out
        self.qkv_bias = qkv_bias
        # context_len is the length of the context for the attention mechanism,
        # which is the number of previous tokens to consider when computing the attention scores
        # This is typically set to the maximum sequence length that the model can handle.
        # For example, if the model can handle sequences of length 128,
        # then context_len should be set to 128.
        # This will be used to create a mask for the attention mechanism,
        # which will prevent the model from attending to future tokens in the sequence.
        # This is important for causal attention mechanisms, where the model should only
        # attend to previous tokens and not future tokens.
        # This is typically set to the maximum sequence length that the model can handle.
        self.context_len = context_len

        # Initialize the trainable weight matrices for query, key, and value
        # These matrices will be used to compute the attention scores
        # and the final output of the attention mechanism.
        # The requires_grad=True will allow these matrices to be updated during training.
        # self.Wq = nn.Parameter(torch.rand(d_in, d_out), requires_grad=True)
        # self.Wk = nn.Parameter(torch.rand(d_in, d_out), requires_grad=True)
        # self.Wv = nn.Parameter(torch.rand(d_in, d_out), requires_grad=True)
        # Other way it o use the nn.linear module to create the trainable weight matrices
        self.Wq = torch.nn.Linear(d_in, d_out, bias=qkv_bias)
        self.Wk = torch.nn.Linear(d_in, d_out, bias=qkv_bias)
        self.Wv = torch.nn.Linear(d_in, d_out, bias=qkv_bias)

    # Forward pass of the self-attention mechanism
    # This method will compute the attention scores and the final output
    # based on the input embeddings.
    # The input x is expected to be a tensor of shape (batch_size, sequence_length, d_in)
    # where d_in is the dimension of the input embeddings.
    # The output will be a tensor of shape (batch_size, sequence_length, d_out)
    # where d_out is the dimension of the output embeddings.
    def forward(self, x):
        return self.forward_self_attention(x)
    
    def forward_self_attention(self, x):
        # Compute the query, key, and value matrices by multiplying the input x
        # with the trainable weight matrices Wq, Wk, and Wv respectively.
        # The resulting tensors will have the shape (batch_size, sequence_length, d_out).
        # keys = x @ self.Wk
        # queries = x @ self.Wq
        # values = x @ self.Wv
        # As we are using the nn.Linear module, we can directly call it on the input x
        queries = self.Wq(x)
        keys = self.Wk(x)
        values = self.Wv(x)

        # Calculate the attention scores by taking the dot product of the queries and keys.
        # The attention scores will have the shape (batch_size, sequence_length, sequence_length).
        # The attention scores represent how much focus each token should have on every other token.
        attention_scores = queries @ keys.mT

        # Do the mask operation to prevent the model from attending to future tokens in the sequence.
        # This is important for causal attention mechanisms, where the model should only
        # attend to previous tokens and not future tokens.
        # The mask is a tensor of shape (context_len, context_len) that contains ones
        # in the upper triangular part and zeros in the lower triangular part.

        # Apply the softmax function to the attention scores to obtain the attention weights.
        # The softmax function will normalize the scores so that they sum to 1 across the sequence length dimension.
        # This will give us the attention weights that indicate the importance of each token
        # with respect to every other token.
        # The reason for dividing by the square root of the dimension of the keys is to
        # prevent the softmax from being too sharp when the dimension is large.
        attention_weights = torch.softmax(attention_scores / keys.shape[-1] ** 0.5, dim=-1)

        # Now we can compute the context vector by multiplying the attention weights with the values.
        # The context vector will have the shape (batch_size, sequence_length, d_out).
        # This context vector is the final output of the self-attention mechanism,
        # which is a weighted sum of the values based on the attention weights.
        # The context vector captures the information from all tokens in the sequence
        # while considering the importance of each token as determined by the attention weights.
        # The reason for multiplying values with attention weights is to aggregate the information
        # from all tokens in the sequence, weighted by their importance.
        context = attention_weights @ values

        # Return the context vector as the output of the self-attention mechanism.
        return context
    
    def forward_causal_attention(self, x):
        # Check if the input x has the expected shape
        # The input x should be a tensor of shape (batch_size, sequence_length, d_in)
        # where d_in is the dimension of the input embeddings.
        if len(x.shape) != 3:
            raise ValueError(f"Input x should be a tensor of shape (batch_size, sequence_length, d_in), but got {x.shape}")
        # Extract the batch size, sequence length, and input dimension from the input x
        # The batch size is the number of sequences in the input batch,
        # the sequence length is the number of tokens in each sequennce,
        # and d_in is the dimension of the input embeddings.
        # For example, if the input x has the shape (32, 128, 512),
        # then batch_size will be 32, sequence_length will be 128,
        # and d_in will be 512. d_in is the dimension of the input embeddings.
        # This will be used to compute the attention scores and the final output.
        batch_size, num_tokens, d_in = x.shape

        # Compute the query, key, and value matrices by multiplying the input x
        # with the trainable weight matrices Wq, Wk, and Wv respectively.
        # The resulting tensors will have the shape (batch_size, num_tokens, d_out).
        # keys = x @ self.Wk
        # queries = x @ self.Wq
        # values = x @ self.Wv
        # As we are using the nn.Linear module, we can directly call it on the input x
        queries = self.Wq(x)
        keys = self.Wk(x)
        values = self.Wv(x)

        # The attention scores are computed by taking the dot product of the queries and keys.
        # This will give us a tensor of shape (batch_size, num_tokens, num_tokens).
        # the 1, 2 in keys.transpose is used to transpose the last two dimensions of the keys tensor.
        # This is necessary because the queries tensor has the shape (batch_size, num_tokens, d_out)
        # and the keys tensor has the shape (batch_size, num_tokens, d_out).
        # The dot product will then be computed between the queries and the transposed keys.
        # This will give us a tensor of shape (batch_size, num_tokens, num_tokens).
        # Example: if queries has the shape (32, 128, 512) and keys has the shape (32, 128, 512),
        # then attention_scores will have the shape (32, 128, 128).

        # The reason using 1,2 is, for each query token we want to compute the attention scores
        # with respect to all key tokens in the sequence. As the keys also arranged in the same way,
        # we can use the transpose operation to align the dimensions correctly for the dot product.
        # So after the transpose the columns of keys will represent the tokens,
        # and the rows of queries will represent the tokens, allowing us to compute the attention scores
        # between each query token and all key tokens in the sequence.

        attention_scores = queries @ keys.transpose(1, 2)

        # Do the mask operation to prevent the model from attending to future tokens in the sequence.
        # This is important for causal attention mechanisms, where the model should only
        # attend to previous tokens and not future tokens.
        # The mask is a tensor of shape (context_len, context_len) that contains ones
        # in the upper triangular part and zeros in the lower triangular part.
        attention_scores.masked_fill_(self.mask.bool()[:num_tokens, :num_tokens], -torch.inf)

        # Apply the softmax function to the attention scores to obtain the attention weights.
        # The softmax function will normalize the scores so that they sum to 1 across the sequence length dimension.
        # This will give us the attention weights that indicate the importance of each token
        # with respect to every other token.
        # The reason for dividing by the square root of the dimension of the keys is to
        # prevent the softmax from being too sharp when the dimension is large.
        attention_weights = torch.softmax(attention_scores / keys.shape[-1] ** 0.5, dim=-1)
        
        # Apply dropout to the attention weights to prevent overfitting.
        # The dropout layer is applied after the attention weights are computed,
        # before they are used to compute the context vector.
        # The dropout rate is set to 0.1 by default, which means 10% of the attention weights
        # will be randomly set to zero during training.
        # This helps to improve the generalization of the model.
        
        if self.dropout_percent > 0:
            attention_weights = self.dropout(attention_weights)

        # Now we can compute the context vector by multiplying the attention weights with the values.
        # The context vector will have the shape (batch_size, sequence_length, d_out).
        # This context vector is the final output of the self-attention mechanism,
        # which is a weighted sum of the values based on the attention weights.
        # The context vector captures the information from all tokens in the sequence
        # while considering the importance of each token as determined by the attention weights.
        # The reason for multiplying values with attention weights is to aggregate the information
        # from all tokens in the sequence, weighted by their importance.
        context = attention_weights @ values
        # Return the context vector as the output of the self-attention mechanism.
        return context