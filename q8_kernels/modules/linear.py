import torch
import torch.nn as nn

import q8_kernels.functional as Q8F

from typing import *

def is_16bit(x) -> bool:
    return x.dtype == torch.float16 or x.dtype == torch.bfloat16

class Q8Linear(nn.Module):
    def __init__(self,  
            in_features: int,
            out_features: int,
            bias: bool = True,
            device=None,
        ):
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(out_features, in_features, device=device, dtype=torch.int8), requires_grad=False)
        
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, device=device, dtype=torch.float), requires_grad=False)
        else:
            self.register_parameter("bias", None)

        self.register_buffer("scales", torch.empty(out_features, device=device, dtype=torch.float))

    def forward(self, x, x_scales=None, fuse_gelu=False, out_dtype=None):
        return Q8F.linear.q8_linear(x, self.weight.data, self.bias.data if self.bias is not None else None, x_scales, self.scales, fuse_gelu, out_dtype)


    @classmethod
    def from_linear(cls, linear: nn.Linear, quant_with_hadamard=True):
        assert linear.weight.data.is_cuda, "input linear layer must be in cuda device"
        assert linear.weight.data.dtype == torch.float8_e4m3fn or is_16bit(linear.weight.data)
        layer = cls(linear.in_features, linear.out_features, linear.bias is not None, linear.weight.device)
        if quant_with_hadamard:
            w_quant, w_scale = Q8F.quantizer.quantize(Q8F.fast_hadamard.hadamard_transform(linear.weight.data))
        else:
            w_quant, w_scale = Q8F.quantizer.quantize(linear.weight.data)
            
        layer.weight.data = w_quant
        layer.scales.data = w_scale
        if linear.bias is not None:
            layer.bias.data = linear.bias.data.float()
        return layer

    # https://re-chill.tistory.com/entry/How-to-hook-to-or-cuda

class Q8LinearLora(nn.Module):
    def __init__(self,  
            in_features: int,
            out_features: int,
            r: int,
            bias: bool = True,
            device=None,
            dtype=torch.bfloat16,
        ):
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.r = r
        self.weight = nn.Parameter(torch.empty(out_features, in_features, device=device, dtype=torch.int8), requires_grad=False)
        self.lora_a = nn.Parameter(torch.empty(r, in_features, device=device, dtype=dtype))
        self.lora_b = nn.Parameter(torch.empty(out_features, r, device=device, dtype=dtype))

        nn.init.normal_(self.lora_a, std=1/r)
        nn.init.zeros_(self.lora_b)
        
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, device=device, dtype=torch.float), requires_grad=False)
        else:
            self.register_parameter("bias", None)

        self.register_buffer("scales", torch.empty(out_features, device=device, dtype=torch.float))

    def forward(self, x, x_scales=None, fuse_gelu=False, out_dtype=None):
        return Q8F.linear.q8_linear_lora(x, self.weight.data, self.bias.data if self.bias is not None else None, 
                                         x_scales, self.scales, 
                                         self.lora_a, self.lora_b, 
                                         fuse_gelu, True, 
                                         out_dtype)

    @classmethod
    def from_linear(cls, linear: Union[nn.Linear, Q8Linear], r: int, quant_with_hadamard=True):
        if isinstance(linear, Q8Linear):
            w_quant = linear.weight.data
            w_scale = linear.scales.data
            bias = linear.bias.data if linear.bias is not None else None

            layer = cls(linear.in_features, linear.out_features, r, bias is not None, linear.weight.device)
            layer.weight.data = w_quant
            layer.scales.data = w_scale
            if bias is not None:
                layer.bias.data = bias
            return layer
        else:
            assert linear.weight.data.is_cuda, "input linear layer must be in cuda device"
            assert linear.weight.data.dtype == torch.float8_e4m3fn or is_16bit(linear.weight.data)
            layer = cls(linear.in_features, linear.out_features, r, linear.bias is not None, linear.weight.device)
            if quant_with_hadamard:
                w_quant, w_scale = Q8F.quantizer.quantize(Q8F.fast_hadamard.hadamard_transform(linear.weight.data))
            else:
                w_quant, w_scale = Q8F.quantizer.quantize(linear.weight.data)
                
            layer.weight.data = w_quant
            layer.scales.data = w_scale
            if linear.bias is not None:
                layer.bias.data = linear.bias.data.float()
            return layer


class FP8Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True, use_hadamard: bool = False, device=None):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_features, in_features, device=device, dtype=torch.float8_e4m3fn), requires_grad=False)
        self.use_hadamard = use_hadamard
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, device=device, dtype=torch.float), requires_grad=False)
        else:
            self.register_parameter("bias", None)

        self.register_buffer("scales", torch.empty(out_features, device=device, dtype=torch.float))

    def forward(self, x):
        return Q8F.linear.fp8_linear(x, self.weight.data, self.bias.data if self.bias is not None else None, 
                                     None, self.scales, 
                                     self.use_hadamard,
                                     x.dtype)

    @classmethod
    def from_linear(cls, linear: nn.Linear, force_cuda=True, use_hadamard=False):
        assert linear.weight.data.is_cuda or force_cuda, "input linear layer must be in cuda device"
        assert is_16bit(linear.weight.data)
        layer = cls(linear.in_features, linear.out_features, linear.bias is not None, use_hadamard, linear.weight.device)
        quant_fn = lambda x: Q8F.quantizer.quantize_fp8(Q8F.fast_hadamard.hadamard_transform(x)) if use_hadamard else Q8F.quantizer.quantize_fp8(x)
        w_quant, w_scale = quant_fn(linear.weight.data.cuda() if force_cuda else linear.weight.data)
        layer.weight.data = w_quant
        layer.scales.data = w_scale
        if linear.bias is not None:
            layer.bias.data = linear.bias.data.float()
        return layer
