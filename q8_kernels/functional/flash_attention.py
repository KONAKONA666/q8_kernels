from typing import Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn

from q8_kernels_cuda.flash_attention._C import flash_attention
from .fast_hadamard import hadamard_transform


def is_16bit(x) -> bool:
    return x.dtype == torch.float16 or x.dtype == torch.bfloat16

class FlashAttnFunc(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q,
        k,
        v,
        softmax_scale,
        batch_mask,
        apply_qk_hadamard,
    ):
        if softmax_scale is None:
            softmax_scale = q.shape[-1] ** (-0.5)
        
        head_size_og = q.size(3)
        if head_size_og % 32 != 0:
            q = torch.nn.functional.pad(q, [0, 32 - head_size_og % 32])
            k = torch.nn.functional.pad(k, [0, 32 - head_size_og % 32])
            v = torch.nn.functional.pad(v, [0, 32 - head_size_og % 32])
        
        assert batch_mask is None or (batch_mask.shape[0] == q.shape[0] and batch_mask.ndim == 1)
        assert k.shape[2] == v.shape[2], "v tokens != k tokens"
        

        if v.shape == k.shape:
            v = v.transpose(2,3).contiguous() # b h s d -> b h d s
        
        if v.shape[-1] % 16 != 0:
            v_tokens = v.shape[-1]
            v_tokens_pad = ((v_tokens + 15)//16)*16 - v_tokens
            v = torch.nn.functional.pad(v, (0, v_tokens_pad))
        
        if apply_qk_hadamard:
            q = hadamard_transform(q)
            k = hadamard_transform(k)
            if is_16bit(q) and is_16bit(k):
                q = q.to(torch.float8_e4m3fn)
                k = k.to(torch.float8_e4m3fn)

        o = flash_attention(q, k, v, softmax_scale, batch_mask)
        return o[..., :head_size_og]
    


def flash_attn_func(
    q,
    k,
    v,
    softmax_scale=None,
    batch_mask=None,
    apply_qk_hadamard=False,
) -> torch.Tensor:
    return FlashAttnFunc.apply(q, k, v, softmax_scale, batch_mask, apply_qk_hadamard)

