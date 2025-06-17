import sys

sys.path.append(".")
from univa.models.qwen2p5vl.configuration_univa_qwen2p5vl import UnivaQwen2p5VLConfig
from univa.models.configuration_univa_denoise_tower import UnivaDenoiseTowerConfig
from univa.models.qwen2p5vl.modeling_univa_qwen2p5vl import (
    UnivaQwen2p5VLForConditionalGeneration,
)
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from diffusers import SD3Transformer2DModel, FluxTransformer2DModel
import torch
import os
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--origin_flux_ckpt_path', type=str, default='/mnt/data/checkpoints/black-forest-labs/FLUX.1-dev',
                        help='Path to the original FLUX checkpoint')
    parser.add_argument('--origin_qwenvl_ckpt_path', type=str, default='/mnt/data/checkpoints/Qwen/Qwen2.5-VL-7B-Instruct',
                        help='Path to the QwenVL base model')
    parser.add_argument('--save_path', type=str, default='/mnt/data/checkpoints/UniWorld_V1',
                        help='Path to the save model')

    return parser.parse_args()

args = parse_args()

origin_flux_ckpt_path = args.origin_flux_ckpt_path
origin_qwenvl_ckpt_path = args.origin_qwenvl_ckpt_path
save_path = args.save_path


processor = AutoProcessor.from_pretrained(origin_qwenvl_ckpt_path)

config = UnivaQwen2p5VLConfig.from_pretrained(origin_qwenvl_ckpt_path)

config.denoise_tower = UnivaDenoiseTowerConfig(
    denoiser_type="flux",
    denoise_projector_type="mlp2x_gelu",
    input_hidden_size=config.hidden_size,
    output_hidden_size=4096,
    denoiser_config=f"{origin_flux_ckpt_path}/transformer/config.json",
)
print(config)

#######################################################################################
model = UnivaQwen2p5VLForConditionalGeneration._from_config(
    config, 
    torch_dtype=torch.float32,
    )
print(model.dtype, model.device)

qwenvl = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    origin_qwenvl_ckpt_path,
    torch_dtype=torch.float32,
)
print(qwenvl.dtype, qwenvl.device)
missing_key, unexpected_key = model.load_state_dict(qwenvl.state_dict(), strict=False)
assert all(['denoise_tower' in miss_key for miss_key in missing_key])
assert len(unexpected_key) == 0
#######################################################################################


#######################################################################################

flux = FluxTransformer2DModel.from_pretrained(
    f"{origin_flux_ckpt_path}",
    subfolder="transformer",
    torch_dtype=torch.float32,
)
print(flux.dtype, flux.device)
model.denoise_tower.denoiser = flux
model.save_pretrained(save_path)
#######################################################################################

processor.save_pretrained(save_path)