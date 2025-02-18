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
from diffusers.models import AutoencoderKL
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
import yaml
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from datasets.img_latent_dataset import ImgLatentDataset
from models import Models
from diffusion import create_diffusion

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

def load_config(config_path):
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config

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
    tokenizer = AutoencoderKL.from_pretrained(args.vae).to(device)
    tokenizer = tokenizer.eval()

    train_config = load_config(args.config)
    if 'downsample_ratio' in train_config['vae']:
        downsample_ratio = train_config['vae']['downsample_ratio']
    else:
        downsample_ratio = 8
    latent_size = train_config['data']['image_size'] // downsample_ratio
    use_diffusion = train_config['scheduler']['diffusion'] if 'diffusion' in train_config['scheduler'] else False
    kwargs = dict(
        input_size=latent_size,
        num_classes=train_config['data']['num_classes'],
        use_qknorm=train_config['model']['use_qknorm'],
        use_swiglu=train_config['model']['use_swiglu'] if 'use_swiglu' in train_config['model'] else False,
        use_rope=train_config['model']['use_rope'] if 'use_rope' in train_config['model'] else False,
        use_rmsnorm=train_config['model']['use_rmsnorm'] if 'use_rmsnorm' in train_config['model'] else False,
        in_channels=train_config['model']['in_chans'] if 'in_chans' in train_config['model'] else 4,
        use_checkpoint=train_config['model']['use_checkpoint'] if 'use_checkpoint' in train_config['model'] else False,
        learn_sigma=train_config['diffusion']['learn_sigma'] if use_diffusion and 'learn_sigma' in train_config['diffusion'] else False,
    )
    assert train_config['model']['return_features']
    kwargs.update(dict(return_features=train_config['model']['return_features']))
    model = Models[train_config['model']['model_type']](**kwargs)
    checkpoint = torch.load(train_config['ckpt_path'], map_location='cpu')
    checkpoint = {k.replace('module.', ''): v for k, v in checkpoint['model'].items()}
    model.load_state_dict(checkpoint)
    model = model.to(device).eval()

    
    if use_diffusion:
        diffusion = create_diffusion(
            timestep_respacing="", 
            learn_sigma=train_config['diffusion']['learn_sigma'], 
            diffusion_steps=train_config['diffusion']['diffusion_steps'], 
            )  # default: 1000 steps, linear noise schedule
        
    # Setup data:
    datasets = [
        ImageFolder(args.data_path, transform=img_transform(p_hflip=0.0, img_size=args.image_size)),
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

    assert not train_config['data']['latent_norm']
    latent_multiplier = train_config['data']['latent_multiplier']
    run_images = 0
    saved_files = 0

    t_list, y_list, c_list, x_list, features_list = [], [], [], [], []
    labels = []
    for batch_idx, batch_data in enumerate(zip(*loaders)):
        run_images += batch_data[0][0].shape[0]
        if run_images % 10 == 0 and rank == 0:
            print(f'{datetime.now()} processing {run_images}x{world_size} of {total_data_in_loop} images')
        
        for loader_idx, data in enumerate(batch_data):
            x = data[0]
            y = data[1]  # (N,)
            x = x.to(device)
            y = y.to(device)
            with torch.no_grad():
                x = tokenizer.encode(x).latent_dist.sample() * latent_multiplier  # (N, C, H, W)
                x = x.repeat(args.num_diffusion_steps, 1, 1, 1)
                y = y.repeat(args.num_diffusion_steps, )
                model_kwargs = dict(y=y, clean_x=x)
                if use_diffusion:
                    t = torch.LongTensor(list(range(diffusion.num_timesteps))[::diffusion.num_timesteps//args.num_diffusion_steps]).to(device)
                    if batch_idx == 0 and rank == 0:
                        print('Input: x shape', x.shape, 'y', y.shape, 't', t.shape)
                    loss_dict = diffusion.training_losses(model, x, t, model_kwargs)
                    all_features = loss_dict["cls_logit"]
                    all_features_cpu = [i.detach().cpu() for i in all_features]
                    t, y, c, x, features = all_features_cpu[0], all_features_cpu[1], all_features_cpu[2], all_features_cpu[3], torch.stack(all_features_cpu[4:], dim=1)

            if batch_idx == 0 and rank == 0:
                print('Feature: t shape', t.shape, 'y', y.shape, 'c', c.shape, 'x', x.shape, 'features', features.shape)
            
            if loader_idx == 0:
                t_list.append(t)
                y_list.append(y)
                c_list.append(c)
                x_list.append(x)
                features_list.append(features)
                labels.append(torch.LongTensor(list(range(diffusion.num_timesteps))[::diffusion.num_timesteps//args.num_diffusion_steps]))
            else:
                raise NotImplementedError()

        if len(t_list) == 100 // args.batch_size:
            t_list = torch.cat(t_list, dim=0)
            y_list = torch.cat(y_list, dim=0)
            c_list = torch.cat(c_list, dim=0)
            x_list = torch.cat(x_list, dim=0)
            features_list = torch.cat(features_list, dim=0)
            labels = torch.cat(labels, dim=0)
            save_dict = {
                't': t_list,
                'y': y_list,
                'c': c_list,
                'x': x_list,
                'features': features_list,
                'labels': labels
            }
            for key in save_dict:
                if rank == 0:
                    print(key, save_dict[key].shape, save_dict[key].dtype)
            save_filename = os.path.join(output_dir, f'features_rank{rank:02d}_shard{saved_files:03d}.safetensors')
            save_file(
                save_dict,
                save_filename,
                metadata={'total_size': f'{features_list.shape[0]}', 'dtype': f'{features_list.dtype}', 'device': f'{features_list.device}'}
            )
            if rank == 0:
                print(f'Saved {save_filename}')
            
            t_list, y_list, c_list, x_list, features_list = [], [], [], [], []
            labels = []
            saved_files += 1

    # save remainder latents that are fewer than 10000 images
    if len(t_list) > 0:
        t_list = torch.cat(t_list, dim=0)
        y_list = torch.cat(y_list, dim=0)
        c_list = torch.cat(c_list, dim=0)
        x_list = torch.cat(x_list, dim=0)
        features_list = torch.cat(features_list, dim=0)
        labels = torch.cat(labels, dim=0)
        save_dict = {
            't': t_list,
            'y': y_list,
            'c': c_list,
            'x': x_list,
            'features': features_list,
            'labels': labels
        }
        for key in save_dict:
            if rank == 0:
                print(key, save_dict[key].shape)
        save_filename = os.path.join(output_dir, f'features_rank{rank:02d}_shard{saved_files:03d}.safetensors')
        save_file(
            save_dict,
            save_filename,
            metadata={'total_size': f'{features_list.shape[0]}', 'dtype': f'{features_list.dtype}', 'device': f'{features_list.device}'}
        )
        if rank == 0:
            print(f'Saved {save_filename}')

    # Calculate latents stats
    dist.barrier()
    # if rank == 0:
    #     dataset = ImgLatentDataset(output_dir, latent_norm=True)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default='/path/to/your/data')
    parser.add_argument("--data_split", type=str, default='imagenet_train')
    parser.add_argument("--output_path", type=str, default="/path/to/your/output")
    parser.add_argument("--vae", type=str, default="/path/to/your/vae")
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--num_diffusion_steps", type=int, default=1000)
    parser.add_argument("--config", type=str, default='/path/to/your/config.yaml')
    args = parser.parse_args()
    assert args.batch_size == 1
    main(args)