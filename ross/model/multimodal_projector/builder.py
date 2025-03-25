import torch
import torch.nn as nn
import re
from transformers.models.convnext.modeling_convnext import ConvNextLayerNorm
from ross.model.multimodal_denoiser.denoiser_dit import RossDenoiser


class IdentityMap(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, *args, **kwargs):
        return x

    @property
    def config(self):
        return {"mm_projector_type": 'identity'}



class SimpleConv(nn.Module):
    def __init__(self, conv_depth, mm_hidden_size, hidden_size, patch_size):
        super().__init__()
        modules = [
            # ConvNextLayerNorm(mm_hidden_size, eps=1e-6, data_format="channels_first"), 
            nn.Conv2d(mm_hidden_size, hidden_size, kernel_size=patch_size, stride=patch_size)
                   ]
        for _ in range(1, conv_depth):
            modules.append(nn.GELU())
            modules.append(nn.Conv2d(hidden_size, hidden_size, kernel_size=1, stride=1))
        self.m = nn.Sequential(*modules)

    @torch.compile
    def forward(self, x):
        assert x.ndim == 4
        x = self.m(x)
        x = x.flatten(2).transpose(1, 2)
        return x

def build_vision_projector(config, delay_load=False, **kwargs):
    projector_type = getattr(config, 'mm_projector_type', 'linear')

    if projector_type == 'linear':
        return nn.Linear(config.mm_hidden_size, config.hidden_size)

    if projector_type.startswith("mlp"):
        mlp_gelu_match = re.match(r'^mlp(\d+)x_gelu$', projector_type)
        if mlp_gelu_match:
            mlp_depth = int(mlp_gelu_match.group(1))
            modules = [nn.Linear(config.mm_hidden_size, config.hidden_size)]
            for _ in range(1, mlp_depth):
                modules.append(nn.GELU())
                modules.append(nn.Linear(config.hidden_size, config.hidden_size))
            return nn.Sequential(*modules)


    if projector_type.startswith("conv"):
        conv_gelu_match = re.match(r'^conv(\d+)x_gelu_p(\d+)$', projector_type)
        if conv_gelu_match:
            conv_depth = int(conv_gelu_match.group(1))
            patch_size = int(conv_gelu_match.group(2))
            vision_projector = SimpleConv(conv_depth, config.mm_hidden_size, config.hidden_size, patch_size)
            return vision_projector
        
    if projector_type == 'identity':
        return IdentityMap()

    raise ValueError(f'Unknown projector type: {projector_type}')


def build_inv_projector(config, delay_load=False, **kwargs):
    projector_type = getattr(config, 'mm_inv_projector_type', 'linear')

    if projector_type == 'linear':
        return nn.Linear(config.hidden_size, config.mm_inv_hidden_size)

    if projector_type.startswith("denoiser"):
        vit_match = re.match(r'^denoiser_vit(\d+)x$', projector_type)
        depth = int(vit_match.group(1))

        if depth == 8:
            width = 1280
        elif depth == 12:
            width = 1536
        else:
            width = 1024

        return RossDenoiser(
            x_channel=config.mm_inv_hidden_size,
            z_channel=config.hidden_size,
            embed_dim=width,
            depth=depth,
            timesteps='1000',
            learn_sigma=False,
            n_patches=config.image_embed_len,
        )

    if projector_type.startswith("mlp"):
        mlp_gelu_match = re.match(r'^mlp(\d+)x_gelu$', projector_type)
        if mlp_gelu_match:
            mlp_depth = int(mlp_gelu_match.group(1))
            modules = [nn.Linear(config.hidden_size, config.hidden_size)]
            if mlp_depth > 2:
                for _ in range(1, mlp_depth - 1):
                    modules.append(nn.GELU())
                    modules.append(nn.Linear(config.hidden_size, config.hidden_size))
            modules.append(nn.GELU())
            modules.append(nn.Linear(config.hidden_size, config.mm_inv_hidden_size))
            return nn.Sequential(*modules)
        
    if projector_type == 'identity':
        return IdentityMap()

    raise ValueError(f'Unknown projector type: {projector_type}')
