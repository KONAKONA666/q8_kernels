import torch
import torch.nn as nn

import q8_kernels.functional as Q8F

from .activations import GELU
from .linear import Q8Linear


class FeedForward(nn.Module):
    def __init__(self, dim:int, mult:int=4, bias:bool=True):
        super().__init__()
        self.inner_dim = dim * mult
        self.act = GELU(dim, self.inner_dim, bias=bias)
        self.proj_down = Q8Linear(self.inner_dim, dim, bias=bias)
    def forward(self, x):
        return self.proj_down(self.act(x))