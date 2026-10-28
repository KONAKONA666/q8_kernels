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
    if hasattr(x, 'dtype'):
        return x.dtype == torch.float16 or x.dtype == torch.bfloat16
    else:
        return x == torch.float16 or x == torch.bfloat16

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
        
        ctx.save_for_backward(b, scale_b)
        ctx.fuse_gelu = fuse_gelu
        ctx.out_dtype = out_dtype   
        
        if bias is not None:
            return q8_mm_bias(a, b, bias, scale_a, scale_b, fuse_gelu, out_dtype)
        else:
            return q8_mm(a, b, scale_a, scale_b, fuse_gelu, out_dtype)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        b, scale_b = ctx.saved_tensors
        fuse_gelu = ctx.fuse_gelu
        out_dtype = ctx.out_dtype
        
        grad_output_fp8, grad_output_scales = quantize_fp8(grad_output)
        w_fp8, w_scales = quantize_fp8((b.to(ctx.out_dtype) * scale_b[:, None]).T)
        grad_x = hadamard_transform(fp8_mm(grad_output_fp8, w_fp8, grad_output_scales, w_scales, False, ctx.out_dtype))
        return grad_x, None, None, None, None, None, None, None, None, None


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

        # Keep already-quantized activations on the same code path.  The
        # previous implementation only assigned ``a_quant`` for FP8/16-bit
        # inputs, so an INT8 activation raised UnboundLocalError here.
        a_quant = a
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
            # save_for_backward accepts tensors only.  There is no need to
            # retain the pre-GELU output when the fused activation is off.
            ctx.save_for_backward(a, b, scale_b, lora_a, lora_b)
        return o

        
    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        fuse_gelu = ctx.fuse_gelu
        out_dtype = ctx.out_dtype
        saved_tensors = ctx.saved_tensors
        if fuse_gelu:
            o, a, b, scale_b, lora_a, lora_b = saved_tensors
        else:
            a, b, scale_b, lora_a, lora_b = saved_tensors
        # a: [b, s, h]
        # b: [h, d]
        # lora_a: [r, h]
        # lora_b: [d, r]
        # grad_output: [b, s, d]
        
        # w = (b * scale_b[:, None]).to(out_dtype)
        
        # print(w.shape)
        if fuse_gelu:
            grad_output = gelu_backward(o, grad_output, out_dtype)

        # This is needed in both backward branches.  Previously the
        # use_hadamard=False branch referenced these names before assignment.
        grad_output_fp8, grad_output_scales = quantize_fp8(grad_output)
        grad_out_lora_b  = grad_output @ lora_b # [b, s, d] @ [d, r] -> [b, s, r]
        if ctx.use_hadamard:
            w = hadamard_transform((b * scale_b[:, None]).to(out_dtype)).T
            w_fp8, w_scales = quantize_fp8(w)

            # grad_x = grad_output @ w + grad_out_lora_b @ lora_a
            grad_x = fp8_mm(grad_output_fp8, w_fp8, grad_output_scales, w_scales, False, out_dtype) + grad_out_lora_b @ lora_a
        else:
            w = (b * scale_b[:, None]).to(out_dtype).T
            w_fp8, w_scales = quantize_fp8(w)
            # grad_x = grad_output @ w + grad_out_lora_b @ lora_a
            grad_x =  fp8_mm(grad_output_fp8, w_fp8, grad_output_scales, w_scales, False, out_dtype) + grad_out_lora_b @ lora_a
        
        # Collapse all batch/sequence dimensions so parameter gradients keep
        # the same shapes as lora_a [r, h] and lora_b [d, r].
        flat_a = a.reshape(-1, a.shape[-1])
        flat_grad_out_lora_b = grad_out_lora_b.reshape(-1, grad_out_lora_b.shape[-1])
        grad_lora_a = flat_grad_out_lora_b.transpose(0, 1) @ flat_a

        flat_grad_output = grad_output.reshape(-1, grad_output.shape[-1])
        flat_lora_input = (a @ lora_a.T).reshape(-1, lora_a.shape[0])
        grad_lora_b = flat_grad_output.transpose(0, 1) @ flat_lora_input
        
        return grad_x, None, None, None, None, grad_lora_a, grad_lora_b, None, None, None


class FP8LinearFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a: torch.Tensor, b: torch.Tensor, bias: Optional[torch.Tensor], 
                scale_a: Optional[torch.Tensor], scale_b:Optional[torch.Tensor], 
                out_dtype: Optional[torch.dtype], use_hadamard: bool) -> torch.Tensor:
        assert (is_16bit(a) and scale_a is None) or (a.dtype == torch.float8_e4m3fn and scale_a is not None), "FP8LinearFunc: a dtype missmatch"
        assert (is_16bit(b) and scale_b is None) or (b.dtype == torch.float8_e4m3fn and scale_b is not None), "FP8LinearFunc: b dtype missmatch"
        assert a.shape[-1] == b.shape[-1], "FP8LinearFunc: mnk missmatch"
        assert bias is None or bias.dtype == torch.float, "FP8LinearFunc: bias must be in fp32"
        assert is_16bit(out_dtype), "FP8LinearFunc: out_dtype must be in 16bit, FIXME: LATER MAYBE"
        quant_fn = lambda x: quantize_fp8(hadamard_transform(x)) if use_hadamard else quantize_fp8(x)
        if is_16bit(a):
            a, scale_a = quant_fn(a)
        if is_16bit(b):
            b, scale_b = quant_fn(b)

        mm_func = fp8_mm_bias if bias is not None else fp8_mm
        mm_args = (a, b, bias, scale_a, scale_b, False, out_dtype) if bias is not None else (a, b, scale_a, scale_b, False, out_dtype)
        ctx.save_for_backward(b, scale_b) # TODO: dW, dBias
        ctx.out_dtype = out_dtype
        ctx.use_hadamard = use_hadamard
        
        o = mm_func(*mm_args)
        return o
    
    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # TODO: dW, dBias
        b, scale_b = ctx.saved_tensors
        # is_nan_before = grad_output.isnan().any()
        quant_fn_w = lambda x: quantize_fp8(hadamard_transform(x).T) if ctx.use_hadamard else quantize_fp8(x)
        
        grad_output_fp8, grad_output_scales = quantize_fp8(grad_output)
        w = (b.to(ctx.out_dtype) * scale_b[:, None]).T
        w_fp8, w_scales = quant_fn_w(w)
        grad_x = fp8_mm(grad_output_fp8, w_fp8, grad_output_scales, w_scales, False, ctx.out_dtype)
        # TODO: add backward for W or ...
        # is_nan_after = grad_x.isnan().any()
        # print(f"is_nan_before: {is_nan_before}, is_nan_after: {is_nan_after}")
        # if is_nan_after.item() and not is_nan_before.item():
        #     # print shape grad_out
        #     print(f"grad_output shape: {grad_output.shape}")
        return grad_x, None, None, None, None, None, None, None, None


def fp8_linear(a: torch.Tensor, b: torch.Tensor, bias: Optional[torch.Tensor]=None, 
                   scale_a: Optional[torch.Tensor]=None, scale_b:Optional[torch.Tensor]=None, 
                   use_hadamard:bool=False,
                   out_dtype: Optional[torch.dtype]=None) -> torch.Tensor:
    return FP8LinearFunc.apply(a, b, bias, scale_a, scale_b, out_dtype, use_hadamard)

def q8_linear_lora(a: torch.Tensor, b: torch.Tensor, bias: Optional[torch.Tensor]=None, 
                   scale_a: Optional[torch.Tensor]=None, scale_b:Optional[torch.Tensor]=None, 
                   lora_a: Optional[torch.Tensor]=None, lora_b: Optional[torch.Tensor]=None,
                   fuse_gelu:bool=False, use_hadamard:bool=True, out_dtype: Optional[torch.dtype]=None) -> torch.Tensor:
    return Q8LinearLora.apply(a, b, bias, scale_a, scale_b, lora_a, lora_b, fuse_gelu, use_hadamard, out_dtype)

def q8_linear(a: torch.Tensor, b: torch.Tensor, bias: Optional[torch.Tensor]=None, 
              scale_a: Optional[torch.Tensor]=None, scale_b:Optional[torch.Tensor]=None, 
              fuse_gelu:bool=False, out_dtype: Optional[torch.dtype]=None) -> torch.Tensor:
    return Q8LinearFunc.apply(a, b, bias, scale_a, scale_b, fuse_gelu, out_dtype)



