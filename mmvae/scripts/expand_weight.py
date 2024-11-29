import torch
import sys
import sys
sys.path.append(".")
from causalvideovae.model.vae.modeling_causalvae import CausalVAEModel
from causalvideovae.model.modules import *

origin_path = "/remote-home1/lzj/causal-video-vae-github/results/test"
output_path = "models/latent8_3d"

print("Loading model!")
model = CausalVAEModel.from_pretrained(origin_path)
new_config = model.config.copy()
new_config['z_channels'] = 8
new_config['embed_dim'] = 8
reset_mix_factor = True
print("Building new model")
new_model = CausalVAEModel.from_config(new_config)

ckpt = new_model.state_dict()
old_ckpt = model.state_dict()

for name, parameter in new_model.named_parameters():
    if name not in old_ckpt:
        # ckpt[name] = torch.zeros_like(ckpt[name])
        continue
    shape1 = ckpt[name].shape
    if sum(shape1) == 1:
        if reset_mix_factor:
            ckpt[name] = torch.tensor([0.])
        continue
    shape2 = old_ckpt[name].shape
    slices = tuple(slice(0, s) for s in shape2)
    mu = torch.mean(old_ckpt[name])
    std = torch.std(old_ckpt[name])
    ckpt[name] = torch.empty_like(ckpt[name]).normal_(mean=mu, std=std)
    ckpt[name][slices] = old_ckpt[name]
    
new_model.load_state_dict(ckpt)
new_model.save_pretrained(output_path)
