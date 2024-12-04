import torch
import torch.nn as nn

import q8_kernels.functional as Q8F

class RMSNorm(nn.Module):
    def __init__(self, dim: int, elementwise_affine: bool = True, device=None):
        super().__init__()
        if elementwise_affine:
            self.weight = nn.Parameter(torch.ones(dim, dtype=torch.float, device=device))
        else:
            self.weight = None
    
    def forward(self, x, out_dtype=None):
        return Q8F.rms_norm.rms_norm_8bit(x, self.weight, out_dtype)