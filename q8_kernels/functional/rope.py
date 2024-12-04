from typing import Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn

from q8_kernels_cuda.ops._C import rope

class ROPE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, cos_freqs: torch.Tensor, sin_freqs: torch.Tensor, out_type: Optional[torch.dtype]) -> torch.Tensor:
        assert (x.dtype == torch.float8_e4m3fn or x.dtype == torch.bfloat16) and cos_freqs.dtype == torch.float and sin_freqs.dtype == torch.float
        assert cos_freqs.shape == x.shape and sin_freqs.shape == sin_freqs.shape

        return rope(x, cos_freqs, sin_freqs, out_type)
    
def apply_rope(x: torch.Tensor, cos_freqs: torch.Tensor, sin_freqs: torch.Tensor, out_type: Optional[torch.dtype]=None) -> torch.Tensor:
    return ROPE.apply(x, cos_freqs, sin_freqs, out_type)