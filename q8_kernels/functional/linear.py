from typing import Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn

from q8_kernels_cuda.gemm._C import q8_mm
from q8_kernels_cuda.gemm._C import q8_mm_bias
from q8_kernels_cuda.gemm._C import fp8_mm, fp8_mm_bias

from q8_kernels_cuda.ops._C import gelu_backward, gelu_forward


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


class Q8LinearLora(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a: torch.Tensor, b: torch.Tensor, bias: Optional[torch.Tensor], 
                     scale_a: Optional[torch.Tensor], scale_b:Optional[torch.Tensor], 
                     lora_a: Optional[torch.Tensor], lora_b: Optional[torch.Tensor],
                     fuse_gelu:bool, use_hadamard:bool, 
                     out_dtype: Optional[torch.dtype]
                ) -> torch.Tensor:
        assert ((a.dtype == torch.float8_e4m3fn or is_16bit(a)) and scale_a is None) or (a.dtype == torch.int8 and scale_a is not None), "Q8LinearFuncLora: a dtype missmatch"
        assert ((b.dtype == torch.float8_e4m3fn or is_16bit(b)) and scale_b is None) or (b.dtype == torch.int8 and scale_b is not None), "Q8LinearFuncLora: b dtype missmatch"
        assert a.shape[-1] == b.shape[-1], "Q8LinearFuncLora: mnk missmatch"
        assert bias is None or bias.dtype == torch.float, "Q8LinearFuncLora: bias must be in fp32"
        assert lora_a is not None and lora_b is not None, "Q8LinearFuncLora: lora_a and lora_b must be provided"
        assert is_16bit(lora_a) and is_16bit(lora_b), "Q8LinearFuncLora: lora_a and lora_b must be in 16bit. 8bit not tested, maybe it works"
        assert lora_a.shape[0] == lora_b.shape[-1], "Q8LinearFuncLora: lora_a and lora_b shape missmatch"
        assert out_dtype is not None and (out_dtype == torch.float16 or out_dtype == torch.bfloat16), "Q8LinearFuncLora: out_dtype must be None or float16 or bfloat16. float8 not tested, maybe it works"

        if a.dtype == torch.float8_e4m3fn or is_16bit(a):
            a_quant, scale_a = quantize(hadamard_transform(a))
        if b.dtype == torch.float8_e4m3fn or is_16bit(b):
            b, scale_b = quantize(hadamard_transform(b))
        
        ctx.fuse_gelu = fuse_gelu
        ctx.out_dtype = out_dtype
        ctx.use_hadamard = use_hadamard
        # TODO: make this more efficient by fusing gelu in to lora
        # maybe pass a@lora_a[b, s, r] to mm func and then in epilogue call mma.sync for fp16 with fp32 accumulate
        # r is small so another mma.sync at the endis not a big deal
        mm_func = q8_mm_bias if bias is not None else q8_mm
        mm_args = (a_quant, b, bias, scale_a, scale_b, False, out_dtype) if bias is not None else (a_quant, b, scale_a, scale_b, False, out_dtype)
        lora_y = torch.nn.functional.linear(torch.nn.functional.linear(a, lora_a), lora_b)
        o = mm_func(*mm_args) + lora_y
        
        if fuse_gelu:
            ctx.save_for_backward(o, a, b, scale_b, lora_a, lora_b)
            o = gelu_forward(o, out_dtype)
        else:
            ctx.save_for_backward(None, a, b, scale_b, lora_a, lora_b)
        return o

        
    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        fuse_gelu = ctx.fuse_gelu
        out_dtype = ctx.out_dtype
        saved_tensors = ctx.saved_tensors
        o, a, b, scale_b, lora_a, lora_b = saved_tensors
        # a: [b, s, h]
        # b: [h, d]
        # lora_a: [r, h]
        # lora_b: [d, r]
        # grad_output: [b, s, d]
        if ctx.use_hadamard:
            w_fp8, w_scales = quantize_fp8(hadamard_transform(((b * scale_b[:, None]).to(out_dtype))).T)
        else:
            w_fp8, w_scales = quantize_fp8((b * scale_b[:, None]).T)
        
        if fuse_gelu:
            grad_output = gelu_backward(o, grad_output, out_dtype)
        
        grad_output_fp8, grad_output_scales = quantize_fp8(grad_output)
        grad_out_lora_b  = grad_output @ lora_b # [b, s, d] @ [d, r] -> [b, s, r]
        
        grad_x = fp8_mm(grad_output_fp8, w_fp8, grad_output_scales, w_scales, False, out_dtype) + grad_out_lora_b @ lora_a
        
        grad_lora_a = grad_out_lora_b.transpose(-1, -2) @ a # [b, r, s] @ [b, s, h] -> [b, r, h] : tflops = 2*b*r*h*s 
        grad_lora_b = grad_output.transpose(-1, -2) @ (a @ lora_a.T) # [b, d, s] @ ([b, s, h] @ [h, r]) => [b, d, s] @ [b, s, r] -> [b, d, r] : tflops = 2*b*s*h*r + 2*b*d*s*r
        
        return grad_x, None, None, None, None, grad_lora_a, grad_lora_b, None, None, None


def q8_linear_lora(a: torch.Tensor, b: torch.Tensor, bias: Optional[torch.Tensor]=None, 
                   scale_a: Optional[torch.Tensor]=None, scale_b:Optional[torch.Tensor]=None, 
                   lora_a: Optional[torch.Tensor]=None, lora_b: Optional[torch.Tensor]=None,
                   fuse_gelu:bool=False, use_hadamard:bool=True, out_dtype: Optional[torch.dtype]=None) -> torch.Tensor:
    return Q8LinearLora.apply(a, b, bias, scale_a, scale_b, lora_a, lora_b, fuse_gelu, use_hadamard, out_dtype)

def q8_linear(a: torch.Tensor, b: torch.Tensor, bias: Optional[torch.Tensor]=None, 
              scale_a: Optional[torch.Tensor]=None, scale_b:Optional[torch.Tensor]=None, 
              fuse_gelu:bool=False, out_dtype: Optional[torch.dtype]=None) -> torch.Tensor:
    return Q8LinearFunc.apply(a, b, bias, scale_a, scale_b, fuse_gelu, out_dtype)



