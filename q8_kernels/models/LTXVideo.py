# Adapted from: https://github.com/Lightricks/LTX-Video
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Literal

import torch
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.embeddings import PixArtAlphaTextProjection
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.normalization import AdaLayerNormSingle
from diffusers.utils import BaseOutput, is_torch_version
from diffusers.utils import logging
from torch import nn

from q8_kernels.blocks.ltx_transformer_block import LTXTransformerBlock

logger = logging.get_logger(__name__)

@dataclass
class Transformer3DModelOutput(BaseOutput):
    """
    The output of [`Transformer2DModel`].

    Args:
        sample (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)` or `(batch size, num_vector_embeds - 1, num_latent_pixels)` if [`Transformer2DModel`] is discrete):
            The hidden states output conditioned on the `encoder_hidden_states` input. If discrete, returns probability
            distributions for the unnoised latent pixels.
    """

    sample: torch.FloatTensor


class LTXTransformer3DModel(ModelMixin, ConfigMixin):
    _supports_gradient_checkpointing = True
    @register_to_config
    def __init__(
        self,
        num_attention_heads: int = 16,
        attention_head_dim: int = 88,
        in_channels: Optional[int] = None,
        out_channels: Optional[int] = None,
        num_layers: int = 1,
        cross_attention_dim: Optional[int] = None,
        bias: bool = False,
        norm_elementwise_affine: bool = True,
        caption_channels: int = None,
        use_rope:bool=True,
        positional_embedding_theta: Optional[float] = None,
        positional_embedding_max_pos: Optional[List[int]] = None,
        timestep_scale_multiplier: Optional[float] = None,
    ):
        inner_dim = num_attention_heads * attention_head_dim
        self.patchify_proj = nn.Linear(in_channels, inner_dim, bias=True)
        self.transformer_blocks = nn.ModuleList(
            [
                LTXTransformerBlock(
                    inner_dim,
                    num_attention_heads,
                    attention_head_dim,
                    cross_attention_dim,
                    bias,
                    use_rope,
                    norm_elementwise_affine,
                )
                for d in range(num_layers)
            ]
        )
        self.out_channels = in_channels if out_channels is None else out_channels
        self.norm_out = nn.LayerNorm(inner_dim, elementwise_affine=False, eps=1e-6)
        self.scale_shift_table = nn.Parameter(
            torch.randn(2, inner_dim) / inner_dim**0.5
        )
        self.proj_out = nn.Linear(inner_dim, self.out_channels)
        self.adaln_single = AdaLayerNormSingle(
            inner_dim, use_additional_conditions=False
        )
        self.caption_projection = PixArtAlphaTextProjection(
                in_features=caption_channels, hidden_size=inner_dim
        )
        self.timestep_scale_multiplier = timestep_scale_multiplier
        self.positional_embedding_max_pos = positional_embedding_max_pos
        self.positional_embedding_theta = positional_embedding_theta

    def get_fractional_positions(self, indices_grid):
        fractional_positions = torch.stack(
            [
                indices_grid[:, i] / self.positional_embedding_max_pos[i]
                for i in range(3)
            ],
            dim=-1,
        )
        return fractional_positions

    def precompute_freqs_cis(self, indices_grid, spacing="exp"):
        dtype = torch.float32  # We need full precision in the freqs_cis computation.
        dim = self.inner_dim
        theta = self.positional_embedding_theta

        fractional_positions = self.get_fractional_positions(indices_grid)

        start = 1
        end = theta
        device = fractional_positions.device
        if spacing == "exp":
            indices = theta ** (
                torch.linspace(
                    math.log(start, theta),
                    math.log(end, theta),
                    dim // 6,
                    device=device,
                    dtype=dtype,
                )
            )
            indices = indices.to(dtype=dtype)
        elif spacing == "exp_2":
            indices = 1.0 / theta ** (torch.arange(0, dim, 6, device=device) / dim)
            indices = indices.to(dtype=dtype)
        elif spacing == "linear":
            indices = torch.linspace(start, end, dim // 6, device=device, dtype=dtype)
        elif spacing == "sqrt":
            indices = torch.linspace(
                start**2, end**2, dim // 6, device=device, dtype=dtype
            ).sqrt()

        indices = indices * math.pi / 2

        if spacing == "exp_2":
            freqs = (
                (indices * fractional_positions.unsqueeze(-1))
                .transpose(-1, -2)
                .flatten(2)
            )
        else:
            freqs = (
                (indices * (fractional_positions.unsqueeze(-1) * 2 - 1))
                .transpose(-1, -2)
                .flatten(2)
            )

        cos_freq = freqs.cos().repeat_interleave(2, dim=-1)
        sin_freq = freqs.sin().repeat_interleave(2, dim=-1)
        if dim % 6 != 0:
            cos_padding = torch.ones_like(cos_freq[:, :, : dim % 6])
            sin_padding = torch.zeros_like(cos_freq[:, :, : dim % 6])
            cos_freq = torch.cat([cos_padding, cos_freq], dim=-1)
            sin_freq = torch.cat([sin_padding, sin_freq], dim=-1)
        return cos_freq.to(dtype), sin_freq.to(dtype)
    
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        indices_grid: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        timestep: Optional[torch.LongTensor] = None,
        class_labels: Optional[torch.LongTensor] = None,
        cross_attention_kwargs: Dict[str, Any] = None,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        return_dict: bool = True,
    ):
        hidden_states = self.patchify_proj(hidden_states)
        timestep = self.timestep_scale_multiplier * timestep
        freqs_cis = self.precompute_freqs_cis(indices_grid)
        batch_size = hidden_states.shape[0]
        timestep, embedded_timestep = self.adaln_single(
            timestep.flatten(),
            {"resolution": None, "aspect_ratio": None},
            batch_size=batch_size,
            hidden_dtype=hidden_states.dtype,
        )
        timestep = timestep.view(batch_size, -1, timestep.shape[-1])
        embedded_timestep = embedded_timestep.view(
            batch_size, -1, embedded_timestep.shape[-1]
        )
        batch_size = hidden_states.shape[0]
        encoder_hidden_states = self.caption_projection(encoder_hidden_states)
        encoder_hidden_states = encoder_hidden_states.view(
                batch_size, -1, hidden_states.shape[-1]
        )
        for block in self.transformer_blocks:
            hidden_states = block(
                    hidden_states,
                    freqs_cis=freqs_cis,
                    attention_mask=attention_mask,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                    timestep=timestep,     
             )
        scale_shift_values = (
            self.scale_shift_table[None, None] + embedded_timestep[:, :, None]
        )
        shift, scale = scale_shift_values[:, :, 0], scale_shift_values[:, :, 1]
        hidden_states = self.norm_out(hidden_states)
        # Modulation
        hidden_states = hidden_states * (1 + scale) + shift
        hidden_states = self.proj_out(hidden_states)
        if not return_dict:
            return (hidden_states,)
        return Transformer3DModelOutput(sample=hidden_states)
    

def get_ltx_transformer():
    return LTXTransformer3DModel(
        32,
        64,
        128,
        128,
        28,
        2048,
        True,
        False,
        2048,
        True,
        10000.0,
        [20, 2048, 2048],
        1000
    )