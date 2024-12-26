from typing import Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn

from q8_kernels_cuda.ops._C import rope, rope_backward

from einops import rearrange

class ROPE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, cos_freqs: torch.Tensor, sin_freqs: torch.Tensor, out_type: Optional[torch.dtype]) -> torch.Tensor:
        assert (x.dtype == torch.float8_e4m3fn or x.dtype == torch.bfloat16) and cos_freqs.dtype == torch.float and sin_freqs.dtype == torch.float
        assert cos_freqs.shape == x.shape and sin_freqs.shape == sin_freqs.shape
        ctx.save_for_backward(cos_freqs, sin_freqs)
        ctx.out_type = out_type
        return rope(x, cos_freqs, sin_freqs, out_type)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        cos_freqs, sin_freqs = ctx.saved_tensors
        out_type = ctx.out_type
        return rope_backward(grad_output, cos_freqs, sin_freqs, out_type), None, None, None

def apply_rope(x: torch.Tensor, cos_freqs: torch.Tensor, sin_freqs: torch.Tensor, out_type: Optional[torch.dtype]=None) -> torch.Tensor:
    
    def apply_rotary_emb(
        input_tensor: torch.Tensor,
        freqs_cis: Tuple[torch.FloatTensor, torch.FloatTensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        cos_freqs = freqs_cis[0]
        sin_freqs = freqs_cis[1]

        t_dup = rearrange(input_tensor, "... (d r) -> ... d r", r=2)
        t1, t2 = t_dup.unbind(dim=-1)
        t_dup = torch.stack((-t2, t1), dim=-1)
        input_tensor_rot = rearrange(t_dup, "... d r -> ... (d r)")

        out = input_tensor * cos_freqs + input_tensor_rot * sin_freqs

        return out
    
    return apply_rotary_emb(x, (cos_freqs, sin_freqs))
    # return ROPE.apply(x, cos_freqs, sin_freqs, out_type)
