# Copyright 2024 Black Forest Labs and The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from dataclasses import dataclass
from typing import Optional

from einops import rearrange
import torch
import torch.nn as nn
from einops.layers.torch import Rearrange



from diffusers.models.embeddings import Timesteps, TimestepEmbedding
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin
from diffusers.utils import BaseOutput


@dataclass
class ReduxImageEncoderOutput(BaseOutput):
    image_embeds: Optional[torch.Tensor] = None



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
        self.conv1 = nn.Conv2d(dim, 2 * dim, kernel_size=3, padding=1, bias=True)
        self.act   = nn.SiLU()
        self.conv2 = nn.Conv2d(2 * dim, dim, kernel_size=3, padding=1, bias=True)

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