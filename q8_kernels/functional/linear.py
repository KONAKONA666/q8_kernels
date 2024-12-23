from typing import Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn

from q8_kernels_cuda.gemm._C import q8_mm
from q8_kernels_cuda.gemm._C import q8_mm_bias

from .fast_hadamard import hadamard_transform
from .quantizer import quantize, quantize_fp8

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



# class Q8LinearLora(torch.autograd.Function):
#     @staticmethod
#     def forward(ctx, a: torch.Tensor, b: torch.Tensor, bias: Optional[torch.Tensor], 
#                      scale_a: Optional[torch.Tensor], scale_b:Optional[torch.Tensor], 
#                      lora_a: Optional[torch.Tensor], lora_b: Optional[torch.Tensor],
#                      fuse_gelu:bool, out_dtype: Optional[torch.dtype]
#                 ) -> torch.Tensor:
#         assert ((a.dtype == torch.float8_e4m3fn or is_16bit(a)) and scale_a is None) or (a.dtype == torch.int8 and scale_a is not None), "Q8LinearFuncLora: a dtype missmatch"
#         assert ((b.dtype == torch.float8_e4m3fn or is_16bit(b)) and scale_b is None) or (b.dtype == torch.int8 and scale_b is not None), "Q8LinearFuncLora: b dtype missmatch"
#         assert a.shape[-1] == b.shape[-1], "Q8LinearFuncLora: mnk missmatch"
#         assert bias is None or bias.dtype == torch.float, "Q8LinearFuncLora: bias must be in fp32"
#         assert lora_a is not None and lora_b is not None, "Q8LinearFuncLora: lora_a and lora_b must be provided"
#         assert is_16bit(lora_a) and is_16bit(lora_b), "Q8LinearFuncLora: lora_a and lora_b must be in 16bit. 8bit not tested, maybe it works"
#         assert lora_a.shape[0] == lora_b.shape[-1], "Q8LinearFuncLora: lora_a and lora_b shape missmatch"
#         assert out_dtype is not None and (out_dtype == torch.float16 or out_dtype == torch.bfloat16), "Q8LinearFuncLora: out_dtype must be None or float16 or bfloat16. float8 not tested, maybe it works"

#         if a.dtype == torch.float8_e4m3fn or is_16bit(a):
#             a_quant, scale_a = quantize(hadamard_transform(a))
#         if b.dtype == torch.float8_e4m3fn or is_16bit(b):
#             b, scale_b = quantize(hadamard_transform(b))
        
#         ctx.fuse_gelu = fuse_gelu
#         ctx.out_dtype = out_dtype

#         if bias is not None:
#             y = q8_mm_bias(a_quant, b, bias, scale_a, scale_b, False, out_dtype)
#             lora_y = torch.functional.linear(torch.functional.linear(a, lora_a), lora_b)
#             o = y + lora_y
#             if fuse_gelu:
#                 ctx.save_for_backward(o, a, b, scale_b, lora_a, lora_b)
#                 o = torch.nn.functional.gelu(o, approximate='tanh')
#             else:
#                 ctx.save_for_backward(None, a, b, scale_b, lora_a, lora_b)
#             return o
#         else:    
#             lora_y = torch.functional.linear(torch.functional.linear(a, lora_a), lora_b)
#             o = q8_mm(a_quant, b, scale_a, scale_b, False, out_dtype) + lora_y
#             if fuse_gelu:
#                 ctx.save_for_backward(o, a, b, scale_b, lora_a, lora_b)
#                 o = torch.nn.functional.gelu(o, approximate='tanh')
#             else:
#                 ctx.save_for_backward(None, a, b, scale_b, lora_a, lora_b)
#             return o
    
#     @staticmethod
#     def backward(ctx, grad_output: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
#         fuse_gelu = ctx.fuse_gelu
#         out_dtype = ctx.out_dtype
#         saved_tensors = ctx.saved_tensors
#         o, a, b, scale_b, lora_a, lora_b = saved_tensors
        
#         w_fp8, w_scales = quantize_fp8((b * scale_b[:, None]).T)
        

#         if fuse_gelu:
#             grad_output = torch.nn.functional.gelu_backward(grad_output, o, approximate='tanh')

#         return grad_output, None, None, None, None, None, None, None, None



        

def q8_linear(a: torch.Tensor, b: torch.Tensor, bias: Optional[torch.Tensor]=None, scale_a: Optional[torch.Tensor]=None, scale_b:Optional[torch.Tensor]=None, fuse_gelu:bool=False, out_dtype: Optional[torch.dtype]=None) -> torch.Tensor:
    return Q8LinearFunc.apply(a, b, bias, scale_a, scale_b, fuse_gelu, out_dtype)



