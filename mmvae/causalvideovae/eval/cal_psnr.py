import numpy as np
import torch
from tqdm import tqdm
from einops import rearrange

def calculate_psnr(video_recon, inputs, device=None):
    video_recon = rearrange(video_recon, "b c t h w -> (b t) c h w").contiguous()
    inputs = rearrange(inputs, "b c t h w -> (b t) c h w").contiguous()
    mse = torch.mean(torch.square(inputs - video_recon), dim=(1,2,3))
    psnr = 20 * torch.log10(1 / torch.sqrt(mse))
    psnr = psnr.mean().detach()
    if psnr == torch.inf:
        return 100
    return psnr.cpu().item()