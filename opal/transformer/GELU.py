import torch
import math
import torch.nn as nn

# The GELU (Gaussian Error Linear Unit) activation function is a smooth, non-monotonic
# activation function that has been shown to work well in deep learning models.
# It is defined as:
# GELU(x) = 0.5 * x * (1 + tanh(sqrt(2 / pi) * (x + 0.044715 * x^3)))
# This function is used to introduce non-linearity in the model, allowing it to learn
# complex patterns in the data. The GELU activation function is often used in transformer
# models and has been shown to outperform other activation functions like ReLU and Swish
# in certain tasks.
# The GELU activation function is differentiable, which is important for training neural networks.
# It is also computationally efficient, making it suitable for large-scale models.
class GELU(nn.Module):
    def __init__(self):
        """
        This is the constructor for the GELU activation function class. It is
        a simple class that inherits from nn.Module and has no parameters.
        """
        super().__init__()
        # Precompute the constant sqrt(2/pi) as a Python float.
        # Using math.sqrt ensures it is treated as a constant in ONNX export,
        # avoiding TracerWarning during torch.onnx.export().
        self.sqrt_2_over_pi = math.sqrt(2.0 / math.pi)

    def forward(self, x):
        """
        This is the forward pass of the GELU activation function. It takes
        the input tensor x and applies the GELU activation function to it.
        The GELU activation function is defined as:
        GELU(x) = 0.5 * x * (1 + tanh(sqrt(2 / pi) * (x + 0.044715 * x^3)))
        The GELU activation function is differentiable, which is important for
        training neural networks. It is also computationally efficient, making
        it suitable for large-scale models.
        """
        return 0.5 * x * (1 + torch.tanh(
            self.sqrt_2_over_pi * (x + 0.044715 * torch.pow(x, 3))
        ))