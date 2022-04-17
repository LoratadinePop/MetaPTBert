"""
    Defines the output class for the MetaPrefixEncoder's parameters.
"""
from turtle import forward
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
        self.embedding_dim = kwargs["embedding_dim"] # the dimension of HyperNet's input
        self.weight_generator = nn.Linear(self.embedding_dim, self.input_dim * self.output_dim)
        self.bias_generator = nn.Linear(self.embedding_dim, self.output_dim)
    
    def forward(self, embeddings):
        weight = self.weight_generator(embeddings).view(self.output_dim, self.input_dim)
        bias = self.bias_generator(embeddings).view(-1)
        return LinearWeight(weight=weight, bias=bias)

class MetaPrefixEncoderHyperNet(nn.Module):
    """
        This module generates the weights for layer-wise meta prefix encoder
        given the layer embeddings. Used in conjunction with torch.nn.linear(input, weight, bias=None),
        thus do not support batch operation. Only ONE weight and bias can be generated once.
    """

    def __init__(self, **kwargs):
        super(MetaPrefixEncoderHyperNet, self).__init__()
        self.input_dim = kwargs["input_dim"]
        self.hidden_dim = kwargs["hidden_dim"]
        self.output_dim = kwargs["output_dim"]
        self.embedding_dim = kwargs["embedding_dim"]
        # Generate the weights for the down projector of MetaPrefixEncoder
        self.down_hypernet = BaseHyperNet(input_dim=self.input_dim, output_dim=self.hidden_dim, embedding_dim=self.embedding_dim)
        # Generate the weights for the up projector of MetaPrefixEncoder
        self.up_hypernet = BaseHyperNet(input_dim=self.hidden_dim, output_dim=self.output_dim, embedding_dim=self.embedding_dim)
        
    def forward(self, layer_embeddings):
        down = self.down_hypernet(layer_embeddings)
        up = self.up_hypernet(layer_embeddings)
        return MetaPrefixEncoderWeight(down_projection=down, up_projection=up)


class MetaPrefixEncoderHyperNetMatMul(nn.Module):
    """
        This module directly generate weight matrix for *torch.matmul()* given the embedding.
        Thus support parallel computing manually, as *torch.nn.functional.linear(input, weight, bias)* only support one weight and bias.
    """
    def __init__(self, **kwargs):
        super().__init__()
        self.input_dim = kwargs["input_dim"]
        self.hidden_dim = kwargs["hidden_dim"]
        self.output_dim = kwargs["output_dim"]
        self.embedding_dim = kwargs["embedding_dim"]
        
        self.down_weight_hypernet = nn.Linear(self.embedding_dim, self.input_dim * self.hidden_dim)
        self.down_bias_hypernet = nn.Linear(self.embedding_dim, self.hidden_dim)
        self.up_weight_hypernet = nn.Linear(self.embedding_dim, self.hidden_dim * self.output_dim)
        self.up_bias_hypernet = nn.Linear(self.embedding_dim, self.output_dim)
    
    def forward(self, embedding):
        return (
            self.down_weight_hypernet(embedding),
            self.down_bias_hypernet(embedding),
            self.up_weight_hypernet(embedding),
            self.up_bias_hypernet(embedding)
        )