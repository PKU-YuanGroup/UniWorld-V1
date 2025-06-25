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
    
class Rearrange(nn.Module):
    def __init__(self, pattern, any_dict):
        super().__init__()
        self.pattern = pattern
        self.any_dict = any_dict
    def forward(self, x):
        return rearrange(x, self.pattern, **self.any_dict)


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
            Rearrange('b (h h2) (w w2) c -> b h w (c h2 w2)', {'h2': 2, 'w2': 2}), # pixunshuffle
            nn.Linear(
                txt_in_features * 4, 
                txt_in_features, 
            ),
            ZeroResBlock(txt_in_features), 
        )

    def forward(self, x: torch.Tensor) -> ReduxImageEncoderOutput:
        projected_x = self.siglip_projector(x)
        return ReduxImageEncoderOutput(image_embeds=projected_x)