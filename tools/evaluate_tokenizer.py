"""
Evaluate tokenizer performance by computing reconstruction metrics.

Metrics include:
- rFID (Reconstruction FID)
- PSNR (Peak Signal-to-Noise Ratio) 
- LPIPS (Learned Perceptual Image Patch Similarity)
- SSIM (Structural Similarity Index)

by Jingfeng Yao
from HUST-VL
"""

import os
import torch
import numpy as np
from tqdm import tqdm
from PIL import Image
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, DistributedSampler
from tools.calculate_fid import calculate_fid_given_paths
from concurrent.futures import ThreadPoolExecutor, as_completed
from torchmetrics import StructuralSimilarityIndexMeasure
from models.lpips import LPIPS
from torchvision.datasets import ImageFolder
from torchvision import transforms
from diffusers.models import AutoencoderKL

def print_with_prefix(content, prefix='Tokenizer Evaluation', rank=0):
    if rank == 0:
        print(f"\033[34m[{prefix}]\033[0m {content}")

def save_image(image, filename):
    Image.fromarray(image).save(filename)

def evaluate_tokenizer(config_path, model_type, data_path, output_path):
    # Initialize distributed training
    dist.init_process_group(backend='nccl')
    local_rank = torch.distributed.get_rank()
    torch.cuda.set_device(local_rank)
    device = torch.device(f'cuda:{local_rank}')

    # Load model
    if local_rank == 0:
        print_with_prefix("Loading model...")
    if model_type == 'vavae':
        from tokenizer.vavae import VA_VAE
        model = VA_VAE(config_path).load().model
    elif model_type == 'sdvae':
        model = AutoencoderKL.from_pretrained("path/to/your/sd-vae-ft-ema").to(device)
    elif model_type == 'marvae':
        from tokenizer.marvae import MAR_VAE
        model = MAR_VAE().load().model

    # Image preprocessing
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(256),
        transforms.CenterCrop(256),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    # Create dataset and dataloader
    dataset = ImageFolder(root=data_path, transform=transform)
    distributed_sampler = DistributedSampler(dataset, num_replicas=dist.get_world_size(), rank=local_rank)
    val_dataloader = DataLoader(
        dataset,
        batch_size=8,
        shuffle=False,
        num_workers=4,
        sampler=distributed_sampler
    )

    # Setup output directories
    folder_name = {
        'vavae': os.path.splitext(os.path.basename(config_path))[0],
        'sdvae': 'sdvae',
        'marvae': 'marvae'
    }[model_type]
    
    save_dir = os.path.join(output_path, folder_name, 'decoded_images')
    ref_path = os.path.join(output_path, folder_name, 'ref_images')
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(ref_path, exist_ok=True)

    if local_rank == 0:
        print_with_prefix(f"Output dir: {save_dir}")
        print_with_prefix(f"Reference dir: {ref_path}")

    # Save reference images if needed
    ref_png_files = [f for f in os.listdir(ref_path) if f.endswith('.png')]
    if len(ref_png_files) < 50000:
        total_samples = 0
        for batch in val_dataloader:
            images = batch[0].to(device)
            for j in range(images.size(0)):
                img = torch.clamp(127.5 * images[j] + 128.0, 0, 255).cpu().permute(1, 2, 0).numpy().astype(np.uint8)
                Image.fromarray(img).save(os.path.join(ref_path, f"ref_image_rank_{local_rank}_{total_samples}.png"))
                total_samples += 1
                if total_samples % 100 == 0 and local_rank == 0:
                    print_with_prefix(f"Rank {local_rank}, Saved {total_samples} reference images")
    dist.barrier()

    # Initialize metrics
    lpips_values = []
    ssim_values = []
    lpips = LPIPS().to(device).eval()
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=(-1.0, 1.0)).to(device)

    # Generate reconstructions and compute metrics
    if local_rank == 0:
        print_with_prefix("Generating reconstructions...")
    all_indices = 0
    for batch in val_dataloader:
        images = batch[0].to(device)
        latents = encode_images(model, images, model_type)
        
        with torch.no_grad():
            decoded_images_tensor = {
                'vavae': lambda: model.decode(latents),
                'marvae': lambda: model.decode(latents),
                'sdvae': lambda: model.decode(latents).sample
            }[model_type]()
            
            decoded_images = torch.clamp(127.5 * decoded_images_tensor + 128.0, 0, 255).permute(0, 2, 3, 1).to("cpu", dtype=torch.uint8).numpy()
        
        # Compute metrics
        lpips_values.append(lpips(decoded_images_tensor, images).mean())
        ssim_values.append(ssim_metric(decoded_images_tensor, images))
        
        # Save reconstructions
        for i, img in enumerate(decoded_images):
            save_image(img, os.path.join(save_dir, f"decoded_image_rank_{local_rank}_{all_indices + i}.png"))
            if (all_indices + i) % 100 == 0 and local_rank == 0:
                print_with_prefix(f"Rank {local_rank}, Processed {all_indices + i} images")
        all_indices += len(decoded_images)
    dist.barrier()

    # Aggregate metrics across GPUs
    lpips_values = torch.tensor(lpips_values).to(device)
    ssim_values = torch.tensor(ssim_values).to(device)
    dist.all_reduce(lpips_values, op=dist.ReduceOp.AVG)
    dist.all_reduce(ssim_values, op=dist.ReduceOp.AVG)
    
    avg_lpips = lpips_values.mean().item()
    avg_ssim = ssim_values.mean().item()

    if local_rank == 0:
        # Calculate FID
        print_with_prefix("Computing rFID...")
        fid = calculate_fid_given_paths([ref_path, save_dir], batch_size=50, dims=2048, device=device, num_workers=16)

        # Calculate PSNR
        print_with_prefix("Computing PSNR...")
        psnr_values = calculate_psnr_between_folders(ref_path, save_dir)
        avg_psnr = sum(psnr_values) / len(psnr_values)

        # Print final results
        print_with_prefix(f"Final Metrics:")
        print_with_prefix(f"rFID: {fid:.3f}")
        print_with_prefix(f"PSNR: {avg_psnr:.3f}")
        print_with_prefix(f"LPIPS: {avg_lpips:.3f}")
        print_with_prefix(f"SSIM: {avg_ssim:.3f}")

    dist.destroy_process_group()

def encode_images(model, images, model_type='vavae'):
    with torch.no_grad():
        posterior = {
            'vavae': lambda: model.encode(images),
            'marvae': lambda: model.encode(images),
            'sdvae': lambda: model.encode(images).latent_dist
        }[model_type]()
        return posterior.sample().to(torch.float32)

def decode_to_images(model, z):
    with torch.no_grad():
        images = model.decode(z)
        images = torch.clamp(127.5 * images + 128.0, 0, 255).permute(0, 2, 3, 1).to("cpu", dtype=torch.uint8).numpy()
    return images

def calculate_psnr(original, processed):
    mse = torch.mean((original - processed) ** 2)
    return 20 * torch.log10(255.0 / torch.sqrt(mse)).item()

def load_image(image_path):
    image = Image.open(image_path).convert('RGB')
    return torch.tensor(np.array(image).transpose(2, 0, 1), dtype=torch.float32)

def calculate_psnr_for_pair(original_path, processed_path):
    return calculate_psnr(load_image(original_path), load_image(processed_path))

def calculate_psnr_between_folders(original_folder, processed_folder):
    original_files = sorted(os.listdir(original_folder))
    processed_files = sorted(os.listdir(processed_folder))

    if len(original_files) != len(processed_files):
        print("Warning: Mismatched number of images in folders")
        return []

    with ThreadPoolExecutor() as executor:
        futures = [
            executor.submit(calculate_psnr_for_pair,
                          os.path.join(original_folder, orig),
                          os.path.join(processed_folder, proc))
            for orig, proc in zip(original_files, processed_files)
        ]
        return [future.result() for future in as_completed(futures)]

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, default='tokenizer/configs/vavae_f16d32.yaml')
    parser.add_argument('--model_type', type=str, default='vavae')
    parser.add_argument('--data_path', type=str, default='/path/to/your/imagenet/ILSVRC2012_validation/data')
    parser.add_argument('--output_path', type=str, default='/path/to/your/output')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    evaluate_tokenizer(config_path=args.config_path, model_type=args.model_type, data_path=args.data_path, output_path=args.output_path)