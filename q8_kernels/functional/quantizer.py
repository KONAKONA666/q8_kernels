import torch

from q8_kernels_cuda.quantizer._C import tokenwise_quant
from typing import Tuple


class TokenWiseQuantizer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x) -> torch.Tensor:
        return tokenwise_quant(x)


def quantize(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    return TokenWiseQuantizer.apply(x)