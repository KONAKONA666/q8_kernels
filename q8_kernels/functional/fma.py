from typing import Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn

from q8_kernels_cuda.ops._C import fma_8bit as fma


def multiply_add(x: torch.Tensor, y: torch.Tensor, z: torch.Tensor, scale_add: float=0.0) -> torch.Tensor:
    return fma(x, y, z, scale_add)

