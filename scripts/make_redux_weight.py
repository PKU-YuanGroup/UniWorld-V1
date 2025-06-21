import sys

sys.path.append(".")
from univa.models.qwen2p5vl.configuration_univa_qwen2p5vl import UnivaQwen2p5VLConfig
from univa.models.configuration_univa_denoise_tower import UnivaDenoiseTowerConfig
from univa.models.modeling_univa_denoise_tower_v1_1 import UnivaDenoiseTower_V1_1
from diffusers import SD3Transformer2DModel, FluxTransformer2DModel
import torch
import os
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--origin_flux_ckpt_path', type=str, default='/mnt/data/checkpoints/black-forest-labs/FLUX.1-dev',
                        help='Path to the original FLUX checkpoint')
    parser.add_argument('--save_path', type=str, default='/mnt/data/checkpoints/UniWorld-V1.1-Redux',
                        help='Path to the save model')

    return parser.parse_args()

args = parse_args()

origin_flux_ckpt_path = args.origin_flux_ckpt_path
save_path = args.save_path



config = UnivaDenoiseTowerConfig(
    denoiser_type="flux",
    denoise_projector_type="mlp2x_gelu",
    # input_hidden_size=config.hidden_size,
    output_hidden_size=4096,
    denoiser_config=f"{origin_flux_ckpt_path}/transformer/config.json",
)
print(config)

#######################################################################################
denoise_tower = UnivaDenoiseTower_V1_1._from_config(
    config, 
    torch_dtype=torch.float32,
    )
print(denoise_tower.dtype, denoise_tower.device)

#######################################################################################


#######################################################################################

flux = FluxTransformer2DModel.from_pretrained(
    f"{origin_flux_ckpt_path}",
    subfolder="transformer",
    torch_dtype=torch.float32,
)
print(flux.dtype, flux.device)
denoise_tower.denoiser = flux
denoise_tower.save_pretrained(save_path)
#######################################################################################

print(save_path)