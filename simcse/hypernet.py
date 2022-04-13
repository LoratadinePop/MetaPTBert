"""
    Defines the output class for the MetaPrefixEncoder's parameters.
"""
from audioop import bias
from matplotlib.font_manager import _Weight
import torch
import torch.nn as nn
from dataclasses import dataclass


@dataclass
class LinearWeight:
    """Base class for the weights and biases of a feed forward linear layer."""
    weight: torch.FloatTensor = None
    bias: torch.FloatTensor = None

@dataclass
class MetaPrefixEncoderWeight:
    """Base class for the weights of the MetaPrefixEncoder layer."""
    down_projection: LinearWeight = None
    up_projection: LinearWeight = None

class BaseHyperNet(nn.Module):
    """
        This module generates the weights for a basic FFN block,
        given the meta embeddings.
    """
    def __init__(self, **kwargs):
        super(BaseHyperNet, self).__init__()
        self.input_dim = kwargs["input_dim"]
        self.output_dim = kwargs["output_dim"]
        self.weight_generator = nn.Linear(self.input_dim, self.input_dim * self.output_dim)
        self.bias_generator = nn.Linear(self.input_dim, self.output_dim)
    
    def forward(self, embeddings):
        weight = self.weight_generator(embeddings).view(self.input_dim, self.output_dim)
        bias = self.bias_generator(embeddings).view(-1)
        return LinearWeight(weight=weight, bias=bias)

class MetaPrefixEncoderHyperNet(nn.Module):
    """
        This module generates the weights for meta prefix encoder
        given the meta embeddings.
    """

    def __init__(self, **kwargs):
        super(MetaPrefixEncoderHyperNet, self).__init__()
        self.input_dim = kwargs["input_dim"]
        self.hidden_dim = kwargs["hidden_dim"]
        self.output_dim = kwargs["output_dim"]
        # Todo: Layer-instance-wise layer_embedding, input avg. embeddings
        # or just 