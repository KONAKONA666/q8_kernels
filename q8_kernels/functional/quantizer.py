import torch

from q8_kernels_cuda.quantizer._C import tokenwise_quant
from typing import Tuple


class TokenWiseQuantizer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, out_type) -> Tuple[torch.Tensor, torch.Tensor]:
        return tokenwise_quant(x, out_type)


def quantize(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    return TokenWiseQuantizer.apply(x, None)

def quantize_fp8(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    return TokenWiseQuantizer.apply(x, torch.float8_e4m3fn)

