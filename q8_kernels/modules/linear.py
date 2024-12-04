import torch
import torch.nn as nn

import q8_kernels.functional as Q8F


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
        
        self.weight = nn.Parameter(torch.empty(out_features, in_features, device=device, dtype=torch.int8), requires_grad=False)
        
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, device=device, dtype=torch.float), requires_grad=False)
        else:
            self.register_parameter("bias", None)

        self.register_buffer("scales", torch.empty(out_features, device=device, dtype=torch.float))

    def forward(self, x, x_scales=None, fuse_gelu=False):
        return Q8F.linear.q8_linear(x, self.weight.data, self.bias.data, x_scales, self.scales, fuse_gelu)


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
    def to(self, *args, **kwargs):
        device, dtype, non_blocking, convert_to_format = torch._C._nn._parse_to(
            *args, **kwargs
        )
        dtype = None
        def convert(t):
            try:
                if convert_to_format is not None and t.dim() in (4, 5):
                    return t.to(
                        device,
                        dtype if t.is_floating_point() or t.is_complex() else None,
                        non_blocking,
                        memory_format=convert_to_format,
                    )
                return t.to(
                    device,
                    dtype if t.is_floating_point() or t.is_complex() else None,
                    non_blocking,
                )
            except NotImplementedError as e:
                if str(e) == "Cannot copy out of meta tensor; no data!":
                    raise NotImplementedError(
                        f"{e} Please use torch.nn.Module.to_empty() instead of torch.nn.Module.to() "
                        f"when moving module from meta to a different device."
                    ) from None
                else:
                    raise
        return self._apply(convert)

