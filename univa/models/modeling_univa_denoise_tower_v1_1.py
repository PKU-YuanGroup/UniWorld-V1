from univa.models.configuration_univa_denoise_tower import UnivaDenoiseTowerConfig
from transformers.modeling_utils import PreTrainedModel

from typing import Any, Dict, Optional, Tuple, Union
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint_sequential
from torch.nn.attention.flex_attention import flex_attention
import numpy as np
from diffusers import FluxTransformer2DModel, SD3Transformer2DModel
from diffusers.utils import is_torch_version
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.models.attention_processor import Attention
from torch.nn.utils.rnn import pad_sequence
from einops import rearrange
from einops.layers.torch import Rearrange
from .modeling_univa_denoise_tower import UnivaDenoiseTower

from dataclasses import dataclass
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin
from diffusers.utils import BaseOutput

def noop(score, b, h, q_idx, kv_idx):
    return score

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

# class Rearrange(nn.Module):
#     def __init__(self, pattern, any_dict):
#         super().__init__()
#         self.pattern = pattern
#         self.any_dict = any_dict
#     def forward(self, x):
#         return rearrange(x, self.pattern, **self.any_dict)

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
            # Rearrange('b (h h2) (w w2) c -> b h w (c h2 w2)', {'h2': 2, 'w2': 2}), # pixunshuffle
            Rearrange('b (h h2) (w w2) c -> b h w (c h2 w2)', h2=2, w2=2), # pixunshuffle
            nn.Linear(
                txt_in_features * 4, 
                txt_in_features, 
            ),
            ZeroResBlock(txt_in_features), 
        )

        new_linear = self.siglip_projector[4]
        assert isinstance(new_linear, nn.Linear)
        if new_linear.weight.numel() > 0:  # from_pretrained will partition parameters when using deepspeed 
            assert new_linear.weight.shape == (4096, 4096 * 4), f'new_linear.weight.shape: {new_linear.weight.shape}'
            with torch.no_grad():
                new_linear.weight.zero_()
                for j in range(4096):
                    base = j * 4
                    new_linear.weight[j, base + 0] = 0.25
                    new_linear.weight[j, base + 1] = 0.25
                    new_linear.weight[j, base + 2] = 0.25
                    new_linear.weight[j, base + 3] = 0.25
                new_linear.bias.zero_()

    @torch.compile
    def forward(self, x: torch.Tensor) -> ReduxImageEncoderOutput:
        # projected_x = self.siglip_projector(x)
        projected_x = checkpoint_sequential(self.siglip_projector, 1, x, use_reentrant=False)
        return ReduxImageEncoderOutput(image_embeds=projected_x)
    
class UnivaDenoiseTower_V1_1(UnivaDenoiseTower):

    def __init__(self, config: UnivaDenoiseTowerConfig):
        super().__init__(config)
        self.siglip_projector = ReduxImageEncoder()

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
    
