import torch
import torch.nn as nn

from q8_kernels.modules.ffn import FeedForward
from q8_kernels.modules.attention import Attention
from q8_kernels.modules.rms_norm import RMSNorm


class LTXTransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, head_dim, kv_dim, bias=True, use_rope=True, norm_affine=False):
        super().__init__()
        self.norm1 = RMSNorm(dim, norm_affine)
        self.attn1 = Attention(dim, num_heads, head_dim, bias, bias, qk_rms_norm=True, kv_dim=None, use_rope=use_rope)
        self.attn2 = Attention(dim, num_heads, head_dim, bias, bias, qk_rms_norm=True, kv_dim=kv_dim, use_rope=False)
        self.norm2 = RMSNorm(dim, norm_affine)
        self.ff = FeedForward(dim, bias=bias)
        self.scale_shift_table = nn.Parameter(
                torch.randn(6, dim) / dim**0.5
        )

    def forward(self, 
                hidden_states, 
                freqs_cis, 
                attention_mask, 
                encoder_hidden_states, encoder_attention_mask, 
                timestep, 
                non_mm_precision=torch.bfloat16):
        
        num_ada_params = self.scale_shift_table.shape[0]
        ada_values = self.scale_shift_table[None, None] + timestep.reshape(
                hidden_states.shape[0], timestep.shape[1], num_ada_params, -1
        )
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
                    ada_values.unbind(dim=2)
                )
        
        norm_hidden_states = self.norm1(hidden_states, non_mm_precision)
        norm_hidden_states = norm_hidden_states * (1 + scale_msa) + shift_msa
        attn_output = self.attn1(norm_hidden_states, freqs_cis, None, attention_mask, non_mm_precision)
        hidden_states = gate_msa * attn_output + hidden_states
    
        attn_output = self.attn2(
            hidden_states,
            freqs_cis=freqs_cis,
            encoder_hidden_states=encoder_hidden_states,
            attention_mask=encoder_attention_mask,
            non_mm_precision=non_mm_precision
        )
        hidden_states = attn_output + hidden_states

        norm_hidden_states = self.norm2(hidden_states, non_mm_precision)
        norm_hidden_states = norm_hidden_states * (1 + scale_mlp) + shift_mlp
        ff_output = self.ff(norm_hidden_states).to(non_mm_precision)
        hidden_states = gate_mlp * ff_output + hidden_states
        
        return hidden_states
