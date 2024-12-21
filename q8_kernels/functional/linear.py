from typing import Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn

from q8_kernels_cuda.gemm._C import q8_mm
from q8_kernels_cuda.gemm._C import q8_mm_bias

from .fast_hadamard import hadamard_transform
from .quantizer import quantize

def is_16bit(x) -> bool:
    return x.dtype == torch.float16 or x.dtype == torch.bfloat16

class Q8LinearFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a: torch.Tensor, b: torch.Tensor, bias: Optional[torch.Tensor], scale_a: Optional[torch.Tensor], scale_b: Optional[torch.Tensor], fuse_gelu: bool, out_dtype: Optional[torch.dtype]) -> torch.Tensor:
        assert ((a.dtype == torch.float8_e4m3fn or is_16bit(a)) and scale_a is None) or (a.dtype == torch.int8 and scale_a is not None), "Q8LinearFunc: a dtype missmatch"
        assert ((b.dtype == torch.float8_e4m3fn or is_16bit(b)) and scale_b is None) or (b.dtype == torch.int8 and scale_b is not None), "Q8LinearFunc: b dtype missmatch"
        assert a.shape[-1] == b.shape[-1], "Q8LinearFunc: mnk missmatch"
        assert bias is None or bias.dtype == torch.float, "Q8LinearFunc: bias must be in fp32"

        if a.dtype == torch.float8_e4m3fn or is_16bit(a):
            a, scale_a = quantize(hadamard_transform(a))
        if b.dtype == torch.float8_e4m3fn or is_16bit(b):
            b, scale_b = quantize(hadamard_transform(b))
        
        if bias is not None:
            return q8_mm_bias(a, b, bias, scale_a, scale_b, fuse_gelu, out_dtype)
        else:
            return q8_mm(a, b, scale_a, scale_b, fuse_gelu, out_dtype)
        

def q8_linear(a: torch.Tensor, b: torch.Tensor, bias: Optional[torch.Tensor]=None, scale_a: Optional[torch.Tensor]=None, scale_b:Optional[torch.Tensor]=None, fuse_gelu:bool=False, out_dtype: Optional[torch.dtype]=None) -> torch.Tensor:
    return Q8LinearFunc.apply(a, b, bias, scale_a, scale_b, fuse_gelu, out_dtype)
