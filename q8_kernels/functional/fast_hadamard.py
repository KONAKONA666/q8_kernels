#https://github.com/Dao-AILab/fast-hadamard-transform/blob/master/fast_hadamard_transform/fast_hadamard_transform_interface.py

import torch
import torch.nn.functional as F
import math
from typing import Optional

from q8_kernels_cuda.ops._C import fast_hadamard_transform


class HadamardTransformFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, scale=1.0, out_type=None):
        ctx._hadamard_transform_scale = scale
        return fast_hadamard_transform(x, scale, out_type)

    @staticmethod
    def backward(ctx, dout):
        # The Hadamard transform matrix is symmetric, so in the backward pass we multiply by its
        # transpose, which is itself.
        return fast_hadamard_transform(dout, ctx._hadamard_transform_scale), None


def hadamard_transform(x: torch.Tensor, scale: Optional[float] = None, out_type:Optional[torch.dtype]=None) -> torch.Tensor:
    """
    Arguments:
        x: (..., dim)
        scale: float. Multiply the output by this number.
    Returns:
        out: (..., dim)

    Multiply each row of x by the Hadamard transform matrix.
    Equivalent to F.linear(x, torch.tensor(scipy.linalg.hadamard(dim))) * scale.
    If dim is not a power of 2, we implicitly pad x with zero so that dim is the next power of 2.
    """
    if scale is None:
        scale = 1/math.sqrt(x.shape[-1])
    return HadamardTransformFn.apply(x, scale, out_type)