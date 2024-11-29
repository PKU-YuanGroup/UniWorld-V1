import torch
import sys
import re

import safetensors
sys.path.append(".")
from causalvideovae.model import CausalVAEModel

origin_ckpt_path = "/remote-home1/lzj/sd_vae.ckpt"
config_path = "models/sd3_vae/config.json"
output_path = "models/sd3_vae_inflate"
init_method = "tail"

model = CausalVAEModel.from_config(config_path)

if origin_ckpt_path.endswith('ckpt'):
    ckpt = torch.load(origin_ckpt_path, map_location="cpu")['state_dict']
elif origin_ckpt_path.endswith('safetensors'):
    ckpt = {}
    with safetensors.safe_open(origin_ckpt_path, framework="pt") as file:
        for k in file.keys():
            ckpt[k] = file.get_tensor(k)
            print("key", k)
            
for name, module in model.named_modules():
    if "loss" in name:
        continue

    if isinstance(module, torch.nn.Conv3d):
        in_channels = module.in_channels
        out_channels = module.out_channels
        kernel_size = module.kernel_size
        old_name = re.sub(".conv$", "", name)
        if old_name + ".weight" not in ckpt:
            print(old_name + ".weight", "not found")
            continue
        if init_method == "tail":
            shape_2d = ckpt[old_name + ".weight"].shape
            new_weight = torch.zeros(*shape_2d)
            new_weight = new_weight.unsqueeze(2).repeat(1, 1, kernel_size[0], 1, 1)
            middle_idx = kernel_size[0] // 2
            new_weight[:, :, -1, :, :] = ckpt[old_name + ".weight"]
            new_bias = ckpt[old_name + ".bias"]
        elif init_method == "avg":
            new_weight = ckpt[old_name + ".weight"].unsqueeze(2)
            new_weight = new_weight.repeat(1, 1, kernel_size[0], 1, 1) / kernel_size[0]
            new_bias = ckpt[old_name + ".bias"]
            assert new_weight.shape == module.weight.shape
        module.weight.data = new_weight.cpu().float()
        module.bias.data = new_bias.cpu().float()
    elif isinstance(module, torch.nn.GroupNorm):
        old_name = name
        if old_name + ".weight" not in ckpt:
            print(old_name + ".weight", "not found")
            continue
        new_weight = ckpt[old_name + ".weight"]
        new_bias = ckpt[old_name + ".bias"]
        module.weight.data = new_weight.cpu().float()
        module.bias.data = new_bias.cpu().float()
    elif isinstance(module, torch.nn.Conv2d):
        in_channels = module.in_channels
        out_channels = module.out_channels
        kernel_size = module.kernel_size
        old_name = name
        if old_name + ".weight" not in ckpt:
            print(old_name + ".weight", "not found")
            continue
        new_weight = ckpt[old_name + ".weight"]
        new_bias = ckpt[old_name + ".bias"]
        assert new_weight.shape == module.weight.shape
        module.weight.data = new_weight.cpu().float()
        module.bias.data = new_bias.cpu().float()

model.save_pretrained(output_path)