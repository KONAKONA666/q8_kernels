import torch
import torch.nn as nn

import q8_kernels.functional as Q8F

from typing import *

from .rms_norm import RMSNorm
from .linear import Q8Linear

def is_16bit(x) -> bool:
    return x.dtype == torch.float16 or x.dtype == torch.bfloat16

class Attention(nn.Module):
    def __init__(self, 
                 query_dim: int, num_heads: int, head_dim: int, 
                 bias: bool=False, out_bias: bool=False,
                 out_dim: Optional[int]=None,
                 qk_rms_norm: bool = False, 
                 kv_dim: Optional[int] = None,
                 use_rope: bool = False
                ):
        super().__init__()

        self.num_heads = num_heads
        self.head_dim = head_dim
        self.query_dim = query_dim
        self.bias = bias
        self.out_bias = out_bias
        self.inner_dim = self.num_heads * self.head_dim
        self.qk_rms_norm = qk_rms_norm
        if qk_rms_norm:
            self.q_norm = RMSNorm(self.inner_dim)
            self.k_norm = RMSNorm(self.inner_dim)
        self.use_rope = use_rope
        kv_dim = query_dim if kv_dim is None else kv_dim
        out_dim = query_dim if out_dim is None else out_dim

        self.to_q = Q8Linear(query_dim, self.inner_dim, bias=bias)
        self.to_k = Q8Linear(kv_dim, self.inner_dim, bias=bias)
        self.to_v = Q8Linear(kv_dim, self.inner_dim, bias=bias)
        
        self.to_out = Q8Linear(self.inner_dim, out_dim, out_bias)
    
    def forward(self, hidden_states, 
                freqs_cis=None, encoder_hidden_states=None, attention_mask=None, 
                non_mm_precision=torch.bfloat16, mm_out_dtype=None, apply_qk_hadamard=True, attn_type='torch'):
        
        if attention_mask is not None and attention_mask.ndim > 1 and attn_type == 'q8_kernels':
            attention_mask = attention_mask.argmin(-1).squeeze().int()
        
        query = self.to_q(hidden_states, out_dtype=mm_out_dtype)
        if self.qk_rms_norm:
            query = self.q_norm(query, non_mm_precision)
        
        if encoder_hidden_states is not None:
            key = self.to_k(encoder_hidden_states, out_dtype=mm_out_dtype)
            if self.qk_rms_norm:
                key = self.k_norm(key, non_mm_precision)
            value = self.to_v(encoder_hidden_states, out_dtype=mm_out_dtype)
        else:
            key = self.to_k(hidden_states, out_dtype=mm_out_dtype)
            if self.qk_rms_norm:
                key = self.k_norm(key, non_mm_precision)
            value = self.to_v(hidden_states, out_dtype=mm_out_dtype)
            if self.use_rope:
                query = Q8F.rope.apply_rope(query, freqs_cis[0], freqs_cis[1], out_type=non_mm_precision)
                key = Q8F.rope.apply_rope(key, freqs_cis[0], freqs_cis[1], out_type=non_mm_precision)
        
        batch_size = query.shape[0]
        query = query.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        if attn_type == 'q8_kernels':
            hidden_states = Q8F.flash_attention.flash_attn_func(query, key, value, 
                                                                batch_mask=attention_mask, 
                                                                apply_qk_hadamard=apply_qk_hadamard)
        elif attn_type == 'torch':
            # attention mask: [batch] -> [batch, heads, 1, key]
            
            query = query.to(non_mm_precision)
            key = key.to(non_mm_precision)
            value = value.to(non_mm_precision)
            
            if attention_mask is not None:
                # Create position indices tensor [batch_size, seq_len]
                positions = torch.arange(key.shape[2], device=query.device).expand(batch_size, -1)
                # Compare with attention_mask [batch_size, 1] to create boolean mask
                mask = positions < attention_mask.unsqueeze(-1)
                # Convert to float mask with -inf for False values
                _attention_mask = torch.where(mask, 0.0, float('-inf'))
                # Reshape to [batch_size, 1, 1, seq_len] and expand to all heads
                attention_mask = _attention_mask.unsqueeze(1).unsqueeze(1).expand(-1, self.num_heads, -1, -1).to(query.dtype)

            hidden_states = torch.nn.functional.scaled_dot_product_attention(query, key, value, 
                                                                            attn_mask=attention_mask, 
                                                                            dropout_p=0.0, is_causal=False)
        else:
            raise ValueError(f"Invalid attention type: {attn_type}")
        hidden_states = hidden_states.transpose(1, 2).reshape(
            batch_size, -1, self.num_heads * self.head_dim
        ).contiguous()

        hidden_states = self.to_out(hidden_states, out_dtype=mm_out_dtype)
        return hidden_states.to(non_mm_precision)