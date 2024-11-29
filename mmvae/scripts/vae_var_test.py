import argparse
from tqdm import tqdm
import torch
import sys
from torch.utils.data import DataLoader, Subset
import os
sys.path.append(".")
from causalvideovae.model import *
from causalvideovae.dataset.video_dataset import ValidVideoDataset
from causalvideovae.utils.video_utils import custom_to_video
from accelerate import Accelerator

@torch.no_grad()
def main(args: argparse.Namespace):
    accelerator = Accelerator()
    
    real_video_dir = args.real_video_dir
    sample_rate = args.sample_rate
    resolution = args.resolution
    crop_size = args.crop_size
    num_frames = args.num_frames
    sample_rate = args.sample_rate
    batch_size = args.batch_size
    num_workers = args.num_workers
    subset_size = args.subset_size
    
    data_type = torch.bfloat16
    
    # ---- Load Model ----
    model_cls = ModelRegistry.get_model(args.model_name)
    vae = model_cls.from_pretrained(args.from_pretrained)
    vae = vae.to(accelerator.device).to(data_type)
    vae.eval()
    
    # ---- Prepare Dataset ----
    dataset = ValidVideoDataset(
        real_video_dir=real_video_dir,
        num_frames=num_frames,
        sample_rate=sample_rate,
        crop_size=crop_size,
        resolution=resolution,
    )
    if subset_size:
        indices = range(subset_size)
        dataset = Subset(dataset, indices=indices)
        
    dataloader = DataLoader(
        dataset, batch_size=batch_size, pin_memory=False, num_workers=num_workers, shuffle=False
    )
    dataloader = accelerator.prepare(dataloader)

    # ---- Inference ----
    latents_list = []
    mean_list = []
    std_list = []
    for batch in tqdm(dataloader, disable=not accelerator.is_local_main_process):
        x = batch['video']
        shift = torch.tensor(vae.config.shift)[None, :, None, None, None].to(device=accelerator.device, dtype=data_type)
        scale = torch.tensor(vae.config.scale)[None, :, None, None, None].to(device=accelerator.device, dtype=data_type)
        x = x.to(device=accelerator.device, dtype=data_type)  # b c t h w
        x = x * 2 - 1
        latents = (vae.encode(x)[0].sample() - shift) * scale
        # latents = vae.encode(x)[0].sample()
        # means = torch.mean(latents)
        # stds = torch.std(latents)
        latents = accelerator.gather(latents)
        if accelerator.is_main_process:
            latents_list.append(latents.cpu())
        # mean_list.append(means)
        # std_list.append(stds)
    
    # print("mean:", torch.mean(means))
    # print("std:", torch.mean(stds))
    if accelerator.is_main_process:
        all_latents_tensor = torch.cat(latents_list)
        means = torch.mean(all_latents_tensor, dim=(0,2,3,4))
        stds = torch.std(all_latents_tensor, dim=(0,2,3,4))
        print("mean:",  means)
        print("std:", stds)
        print(f'normalizer={1/stds}')
    
    # normalizer = 1 / std
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--real_video_dir", type=str, default="")
    parser.add_argument("--from_pretrained", type=str, default="")
    parser.add_argument("--sample_fps", type=int, default=30)
    parser.add_argument("--resolution", type=int, default=336)
    parser.add_argument("--crop_size", type=int, default=None)
    parser.add_argument("--num_frames", type=int, default=17)
    parser.add_argument("--sample_rate", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--subset_size", type=int, default=None)
    parser.add_argument("--model_name", type=str, default=None, help="")
    parser.add_argument("--device", type=str, default="cuda")

    args = parser.parse_args()
    main(args)
    
