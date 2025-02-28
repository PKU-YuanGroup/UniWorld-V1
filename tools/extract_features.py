# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# LightningDiT: https://github.com/hustvl/LightningDiT
# --------------------------------------------------------
import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision.datasets import ImageFolder
import argparse
import os
from safetensors.torch import save_file
from datetime import datetime
import torch
import math
import random
import numpy as np
from PIL import Image
from torchvision import transforms
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from datasets.img_latent_dataset import ImgLatentDataset
from tokenizer import VAE_Models



def random_crop_arr(pil_image, image_size, min_crop_frac=0.8, max_crop_frac=1.0):
    min_smaller_dim_size = math.ceil(image_size / max_crop_frac)
    max_smaller_dim_size = math.ceil(image_size / min_crop_frac)
    smaller_dim_size = random.randrange(min_smaller_dim_size, max_smaller_dim_size + 1)

    # We are not on a new enough PIL to support the `reducing_gap`
    # argument, which uses BOX downsampling at powers of two first.
    # Thus, we do it by hand to improve downsample quality.
    while min(*pil_image.size) >= 2 * smaller_dim_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = smaller_dim_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = random.randrange(arr.shape[0] - image_size + 1)
    crop_x = random.randrange(arr.shape[1] - image_size + 1)
    return Image.fromarray(arr[crop_y : crop_y + image_size, crop_x : crop_x + image_size])


def center_crop_arr(pil_image, image_size):
    """
    Center cropping implementation from ADM.
    https://github.com/openai/guided-diffusion/blob/8fb3ad9197f16bbc40620447b2742e13458d2831/guided_diffusion/image_datasets.py#L126
    """
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size])

def img_transform(p_hflip=0, img_size=256):
    """Image preprocessing transforms
    Args:
        p_hflip: Probability of horizontal flip
        img_size: Target image size, use default if None
    Returns:
        transforms.Compose: Image transform pipeline
    """
    img_transforms = [
        transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, img_size)),
        transforms.RandomHorizontalFlip(p=p_hflip),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
    ]
    return transforms.Compose(img_transforms)


class ImageFolderWithPath(ImageFolder):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_len = 64

    def __getitem__(self, index):
        path = self.samples[index][0]
        path = '/'.join(path.split('/')[-2:])
        encoded_path = path.encode("utf-8")
        assert len(encoded_path) < self.max_len
        paths_tensor = torch.zeros((self.max_len, ), dtype=torch.uint8)
        paths_tensor[:len(encoded_path)] = torch.tensor(list(encoded_path), dtype=torch.uint8)

        paths_recovered = "".join(map(chr, paths_tensor.tolist())).rstrip("\x00")
        assert paths_recovered == path

        return super().__getitem__(index) + (paths_tensor, )
    
def main(args):
    """
    Run a tokenizer on full dataset and save the features.
    """
    assert torch.cuda.is_available(), "Extract features currently requires at least one GPU."

    # Setup DDP:
    try:
        dist.init_process_group("nccl")
        rank = dist.get_rank()
        device = rank % torch.cuda.device_count()
        world_size = dist.get_world_size()
        seed = args.seed + rank
        if rank == 0:
            print(f"Starting rank={rank}, seed={seed}, world_size={world_size}.")
    except Exception as e:
        print(f"Failed to initialize DDP. Running in local mode. Error: {e}")
        rank = 0
        device = 0
        world_size = 1
        seed = args.seed
    torch.manual_seed(seed)
    torch.cuda.set_device(device)

    # Setup feature folders:
    output_dir = os.path.join(args.output_path, f'{args.data_split}_{args.image_size}')
    if rank == 0:
        os.makedirs(output_dir, exist_ok=True)

    # Create model:
    tokenizer = VAE_Models[args.vae_type](args.vae_path)

    # Setup data:
    datasets = [
        ImageFolderWithPath(args.data_path, transform=img_transform(p_hflip=0.0, img_size=args.image_size)),
        ImageFolderWithPath(args.data_path, transform=img_transform(p_hflip=1.0, img_size=args.image_size))
    ]
    samplers = [
        DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=False,
            seed=args.seed
        ) for dataset in datasets
    ]
    loaders = [
        DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            sampler=sampler,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=False
        ) for dataset, sampler in zip(datasets, samplers)
    ]
    total_data_in_loop = len(loaders[0].dataset)
    if rank == 0:
        print(f"Total data in one loop: {total_data_in_loop}")

    run_images = 0
    saved_files = 0
    latents = []
    latents_flip = []
    labels = []
    paths = []
    for batch_idx, batch_data in enumerate(zip(*loaders)):
        run_images += batch_data[0][0].shape[0]
        if run_images % 100 == 0 and rank == 0:
            print(f'{datetime.now()} processing ({run_images * world_size}) of {total_data_in_loop} images')
        
        for loader_idx, data in enumerate(batch_data):
            x = data[0]
            y = data[1]  # (N,)
            path = data[2]  # (N,)
            
            x = x.to(device)
            z = tokenizer.encode_images(x).detach().cpu()  # (N, C, H, W)

            if batch_idx == 0 and rank == 0:
                print('latent shape', z.shape, 'dtype', z.dtype)
            
            if loader_idx == 0:
                latents.append(z)
                labels.append(y)
                paths.append(path)
            else:
                latents_flip.append(z)

        if len(latents) == 10000 // args.batch_size:
            latents = torch.cat(latents, dim=0)
            latents_flip = torch.cat(latents_flip, dim=0)
            labels = torch.cat(labels, dim=0)
            paths = torch.cat(paths, dim=0)
            save_dict = {
                'latents': latents,
                'latents_flip': latents_flip,
                'labels': labels,
                'paths': paths
            }
            for key in save_dict:
                if rank == 0:
                    print(key, save_dict[key].shape)
            save_filename = os.path.join(output_dir, f'latents_rank{rank:02d}_shard{saved_files:03d}.safetensors')
            save_file(
                save_dict,
                save_filename,
                metadata={'total_size': f'{latents.shape[0]}', 'dtype': f'{latents.dtype}', 'device': f'{latents.device}'}
            )
            if rank == 0:
                print(f'Saved {save_filename}')
            
            latents = []
            latents_flip = []
            labels = []
            paths = []
            saved_files += 1

    # save remainder latents that are fewer than 10000 images
    if len(latents) > 0:
        latents = torch.cat(latents, dim=0)
        latents_flip = torch.cat(latents_flip, dim=0)
        labels = torch.cat(labels, dim=0)
        paths = torch.cat(paths, dim=0)
        save_dict = {
            'latents': latents,
            'latents_flip': latents_flip,
            'labels': labels,
            'paths': paths
        }
        for key in save_dict:
            if rank == 0:
                print(key, save_dict[key].shape)
        save_filename = os.path.join(output_dir, f'latents_rank{rank:02d}_shard{saved_files:03d}.safetensors')
        save_file(
            save_dict,
            save_filename,
            metadata={'total_size': f'{latents.shape[0]}', 'dtype': f'{latents.dtype}', 'device': f'{latents.device}'}
        )
        if rank == 0:
            print(f'Saved {save_filename}')

    # Calculate latents stats
    dist.barrier()
    if rank == 0:
        dataset = ImgLatentDataset(output_dir, latent_norm=True)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default='/path/to/your/data')
    parser.add_argument("--data_split", type=str, default='imagenet_train')
    parser.add_argument("--output_path", type=str, default="/path/to/your/output")
    parser.add_argument("--vae_type", type=str, default="sdvae")
    parser.add_argument("--vae_path", type=str, default="/path/to/your/vae")
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=40)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=8)
    args = parser.parse_args()
    main(args)