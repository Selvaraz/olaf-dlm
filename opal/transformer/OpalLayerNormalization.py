import torch
import torch.nn as nn

# The purpose of the OpalLayerNormalization class is to apply layer normalization
# to the input tensor. Layer normalization is a technique to normalize the inputs
# across the features for each example in the batch, which helps stabilize and
# accelerate the training of deep neural networks.
class OpalLayerNormalization(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        # This is a small value added to the variance to prevent division by zero
        # during the computation of the normalized input. It is a common
        # technique used in many neural network libraries, such as TensorFlow
        # and PyTorch.
        self.eps = 1e-5
        # The scale and shift parameters are learnable parameters that are
        # initialized to ones and zeros, respectively. The need for these 
        # parameters are to allow the model to learn the optimal scale and shift
        # values for the normalized input.
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))
    def forward(self, x):
        """
        This is the forward pass of the OpalLayerNormalization class. It computes
        the layer normalization of the input tensor x. The layer normalization
        is computed by first computing the mean and variance of the input tensor
        x across the features (i.e. the last dimension of x). Then the normalized
        input x is computed by subtracting the mean and dividing by the square
        root of the variance plus a small value epsilon to prevent division by
        zero. Finally, the normalized input x is scaled and shifted by the
        learnable parameters self.scale and self.shift, respectively.
        """
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift