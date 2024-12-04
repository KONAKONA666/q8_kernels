import torch
import torch.nn as nn

import q8_kernels.functional as Q8F

from .linear import Q8Linear


class GELU(nn.Module):
    def __init__(self, dim_in: int, dim_out: int, approximate: str = "none", bias: bool = True):
        super().__init__()
        self.proj = Q8Linear(dim_in, dim_out, bias=bias)

    def forward(self, hidden_states):
        hidden_states = self.proj(hidden_states, fuse_gelu=True)
        return hidden_states