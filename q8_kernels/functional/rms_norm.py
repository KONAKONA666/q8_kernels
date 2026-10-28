from typing import Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn

from q8_kernels_cuda.ops._C import rms_norm, rms_norm_backward

class RMSNorm8bit(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, weights: Optional[torch.Tensor], out_type: Optional[torch.Tensor]) -> torch.Tensor:
        assert weights is None or weights.dtype == torch.float, "RMSNorm8bit: dtype missmatch"
        x_normed, x_norms =  rms_norm(x, weights, out_type)
        ctx.has_weights = weights is not None
        if ctx.has_weights:
            ctx.save_for_backward(x_normed, x_norms, weights)
        else:
            # save_for_backward accepts tensors only; keep the optional
            # parameter state separately instead of storing None as a tensor.
            ctx.save_for_backward(x_normed, x_norms)
        ctx.out_type = out_type
        return x_normed

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if ctx.has_weights:
            x_normed, x_norms, weights = ctx.saved_tensors
        else:
            x_normed, x_norms = ctx.saved_tensors
            weights = None
        out_type = ctx.out_type 
        return rms_norm_backward(x_normed, x_norms, grad_output, weights, out_type), None, None

def rms_norm_8bit(x: torch.Tensor, weights: Optional[torch.Tensor] = None, out_type: Optional[torch.dtype]=None) -> torch.Tensor:
    return RMSNorm8bit.apply(x, weights, out_type)
