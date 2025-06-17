from univa.models.configuration_univa_denoise_tower import UnivaDenoiseTowerConfig
from transformers.modeling_utils import PreTrainedModel

from typing import Any, Dict, Optional, Tuple, Union
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.attention.flex_attention import flex_attention
import numpy as np
from diffusers import FluxTransformer2DModel, SD3Transformer2DModel
from diffusers.utils import is_torch_version
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.models.attention_processor import Attention
from diffusers.models.embeddings import apply_rotary_emb, Timesteps, TimestepEmbedding
from torch.nn.utils.rnn import pad_sequence
from einops.layers.torch import Rearrange
from einops import rearrange
from .modeling_univa_denoise_tower import UnivaDenoiseTower

from dataclasses import dataclass
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin
from diffusers.utils import BaseOutput

def noop(score, b, h, q_idx, kv_idx):
    return score

class FluxFlexAttnProcessor2_0:
    """Attention processor used typically in processing the SD3-like self-attention projections."""

    def __init__(self):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("FluxAttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0.")

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: torch.FloatTensor = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
    ) -> torch.FloatTensor:
        batch_size, _, _ = hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape

        # `sample` projections.
        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        # the attention in FluxSingleTransformerBlock does not use `encoder_hidden_states`
        if encoder_hidden_states is not None:
            # `context` projections.
            encoder_hidden_states_query_proj = attn.add_q_proj(encoder_hidden_states)
            encoder_hidden_states_key_proj = attn.add_k_proj(encoder_hidden_states)
            encoder_hidden_states_value_proj = attn.add_v_proj(encoder_hidden_states)

            encoder_hidden_states_query_proj = encoder_hidden_states_query_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)
            encoder_hidden_states_key_proj = encoder_hidden_states_key_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)
            encoder_hidden_states_value_proj = encoder_hidden_states_value_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)

            if attn.norm_added_q is not None:
                encoder_hidden_states_query_proj = attn.norm_added_q(encoder_hidden_states_query_proj)
            if attn.norm_added_k is not None:
                encoder_hidden_states_key_proj = attn.norm_added_k(encoder_hidden_states_key_proj)

            # attention
            query = torch.cat([encoder_hidden_states_query_proj, query], dim=2)
            key = torch.cat([encoder_hidden_states_key_proj, key], dim=2)
            value = torch.cat([encoder_hidden_states_value_proj, value], dim=2)

        if image_rotary_emb is not None:

            query = apply_rotary_emb(query, image_rotary_emb)
            key = apply_rotary_emb(key, image_rotary_emb)

        # hidden_states = F.scaled_dot_product_attention(
        #     query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        # )

        hidden_states = flex_attention(
            query, 
            key, 
            value,
            score_mod=noop, 
        )
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        if encoder_hidden_states is not None:
            encoder_hidden_states, hidden_states = (
                hidden_states[:, : encoder_hidden_states.shape[1]],
                hidden_states[:, encoder_hidden_states.shape[1] :],
            )

            # linear proj
            hidden_states = attn.to_out[0](hidden_states)
            # dropout
            hidden_states = attn.to_out[1](hidden_states)

            encoder_hidden_states = attn.to_add_out(encoder_hidden_states)

            return hidden_states, encoder_hidden_states
        else:
            return hidden_states



class PixArtAlphaAspectEmbeddings(nn.Module):
    """
    For PixArt-Alpha.

    Reference:
    https://github.com/PixArt-alpha/PixArt-alpha/blob/0f55e922376d8b797edd44d25d0e7464b260dcab/diffusion/model/nets/PixArtMS.py#L164C9-L168C29
    """

    def __init__(self, embedding_dim=4096):
        super().__init__()

        self.aspect_ratio_proj = Timesteps(num_channels=512, flip_sin_to_cos=True, downscale_freq_shift=0)
        self.aspect_ratio_embedder = TimestepEmbedding(in_channels=512, time_embed_dim=embedding_dim)

        self.silu = nn.SiLU()
        self.linear = nn.Linear(embedding_dim, embedding_dim, bias=True)

    def forward(
        self,
        aspect_ratio: torch.Tensor,
        hidden_dtype: Optional[torch.dtype] = None,
    ):
        aspect_ratio_emb = self.aspect_ratio_proj(aspect_ratio).to(hidden_dtype)
        aspect_ratio_emb = self.aspect_ratio_embedder(aspect_ratio_emb)
        return self.linear(self.silu(aspect_ratio_emb)).unsqueeze(1)


class ZeroResBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        # 1) First conv: normal fan‑in init
        self.conv1 = nn.Conv2d(dim, 2 * dim, kernel_size=3, padding=1, bias=True)
        # 2) Activation
        self.act   = nn.SiLU()
        # 3) Final conv: zero init so branch output starts at 0
        self.conv2 = nn.Conv2d(2 * dim, dim, kernel_size=3, padding=1, bias=True)
        
        # from_pretrained will partition parameters when using deepspeed 
        if self.conv1.weight.numel() > 0 and self.conv2.weight.numel() > 0:
            nn.init.kaiming_normal_(self.conv1.weight, nonlinearity="relu")
            nn.init.zeros_(self.conv1.bias)
            nn.init.zeros_(self.conv2.weight)
            nn.init.zeros_(self.conv2.bias)

    def forward(self, x):
        # assume x is (b, h, w, c); convert to (b, c, h, w)
        x0 = rearrange(x, "b h w c -> b c h w")
        
        # residual branch
        r  = self.conv1(x0)
        r  = self.act(r)
        r  = self.conv2(r)    # starts as zero
        
        # add & return in original layout
        out = x0 + r
        return rearrange(out, "b c h w -> b (h w) c")
    


@dataclass
class ReduxImageEncoderOutput(BaseOutput):
    image_embeds: Optional[torch.Tensor] = None

class ReduxImageEncoder(ModelMixin, ConfigMixin):
    @register_to_config
    def __init__(
        self,
        redux_dim: int = 1152,
        txt_in_features: int = 4096,
        anyres: str='any_11ratio', 
        anchor_pixels: int = 1024 * 1024, 
        stride: int = 32
    ) -> None:
        super().__init__()

        self.siglip_projector = nn.Sequential(
            nn.Linear(
                redux_dim, 
                txt_in_features * 3, 
            ),
            nn.SiLU(),
            nn.Linear(
                txt_in_features * 3, 
                txt_in_features,
            ),
            Rearrange('b (h h2) (w w2) c -> b h w (c h2 w2)', h2=2, w2=2), # pixunshuffle
            nn.Linear(
                txt_in_features * 4, 
                txt_in_features, 
            ),
            ZeroResBlock(txt_in_features), 
        )
        self.aspect_ratio_embedder = PixArtAlphaAspectEmbeddings(txt_in_features)

    def forward(self, x: torch.Tensor, aspect_ratio: Optional[torch.Tensor] = None) -> ReduxImageEncoderOutput:
        projected_x = self.siglip_projector(x)
        if aspect_ratio is None:
            aspect_ratio = torch.full(
                    (projected_x.shape[0],),  # 直接一次性创建所有
                    fill_value=1.0,
                    device=projected_x.device,
                    dtype=torch.float32, 
                )
        aspect_ratio_embed = self.aspect_ratio_embedder(aspect_ratio, projected_x.dtype)
        projected_x = torch.concat([projected_x, aspect_ratio_embed], dim=1)
        return ReduxImageEncoderOutput(image_embeds=projected_x)
    
class UnivaDenoiseTower_V1_1(UnivaDenoiseTower):

    def __init__(self, config: UnivaDenoiseTowerConfig):
        super().__init__(config)
        # self.replace_sdpa_to_flexattn()
        self.siglip_projector = ReduxImageEncoder()

    def replace_sdpa_to_flexattn(self):
        for m in self.denoiser.transformer_blocks:
            m.attn.processor = FluxFlexAttnProcessor2_0()
        for m in self.denoiser.single_transformer_blocks:
            m.attn.processor = FluxFlexAttnProcessor2_0()

    @torch.compile
    def compile_forward(
        self,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        return self.forward(*args, **kwargs)
    
    def forward(
        self,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        return super().forward(*args, **kwargs)
    
