# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DiT: https://github.com/facebookresearch/DiT
# LightningDiT: https://github.com/hustvl/LightningDiT
# --------------------------------------------------------

import os, math, argparse, yaml, torch, numpy as np
from time import strftime
from PIL import Image
from tqdm import tqdm
import torch.distributed as dist
from accelerate import Accelerator
from torchvision import transforms
from torch.utils.data import DataLoader, DistributedSampler
from concurrent.futures import ThreadPoolExecutor, as_completed
from torchmetrics import StructuralSimilarityIndexMeasure
from torchvision.datasets import ImageFolder
# local imports

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from models.lpips import LPIPS
from tools.calculate_fid import calculate_fid_given_paths
from transport import create_transport, Sampler
from tokenizer import VAE_Models
from datasets.img_latent_dataset import ImgLatentDataset

def save_image(image, filename):
    Image.fromarray(image).save(filename)

# sample function
def do_sample(train_config, accelerator, ckpt_path=None, cfg_scale=None, model=None, vae=None, demo_sample_mode=False):
    """
    Run sampling.
    """
    folder_name = f"{train_config['vae']['vae_type'].replace('/', '-')}-ckpt-{ckpt_path.split('/')[-1].split('.')[0]}-{train_config['flowvae_sample']['num_sampling_steps']}".lower()

    if cfg_scale is None:
        cfg_scale = train_config['flowvae_sample']['cfg_scale']
    cfg_interval_start = train_config['flowvae_sample']['cfg_interval_start'] if 'cfg_interval_start' in train_config['flowvae_sample'] else 0
    timestep_shift = train_config['flowvae_sample']['timestep_shift'] if 'timestep_shift' in train_config['flowvae_sample'] else 0
    if cfg_scale > 1.0:
        folder_name += f"-interval{cfg_interval_start:.2f}"+f"-cfg{cfg_scale:.2f}"
        folder_name += f"-shift{timestep_shift:.2f}"
    folder_name += f"_decoded_images"

    if demo_sample_mode:
        cfg_interval_start = 0
        timestep_shift = 0
        cfg_scale = 1.0

    output_dir = train_config['train']['output_dir']
    sample_folder_dir = os.path.join(output_dir, folder_name)
    if accelerator.process_index == 0:
        if not demo_sample_mode:
            print_with_prefix('Sample_folder_dir=', sample_folder_dir)
        print_with_prefix('ckpt_path=', ckpt_path)
        print_with_prefix('cfg_scale=', cfg_scale)
        print_with_prefix('cfg_interval_start=', cfg_interval_start)
        print_with_prefix('timestep_shift=', timestep_shift)

    if not os.path.exists(sample_folder_dir):
        if accelerator.process_index == 0:
            os.makedirs(output_dir, exist_ok=True) 
            if not demo_sample_mode:
                os.makedirs(sample_folder_dir, exist_ok=True) 

    torch.backends.cuda.matmul.allow_tf32 = True  # True: fast but may lead to some small numerical differences
    assert torch.cuda.is_available(), "Sampling with DDP requires at least one GPU. sample.py supports CPU-only usage"
    torch.set_grad_enabled(False)


    # Setup DDP:
    device = accelerator.device
    seed = train_config['train']['global_seed'] * accelerator.num_processes + accelerator.process_index
    torch.manual_seed(seed)
    # torch.cuda.set_device(device)
    print_with_prefix(f"Starting rank={accelerator.local_process_index}, seed={seed}, world_size={accelerator.num_processes}.")
    rank = accelerator.local_process_index

    # Load model:    
    shift_lg = train_config['flowvae_transport']['shift_lg'] if 'shift_lg' in train_config['flowvae_transport'] else True
    shifted_mu = train_config['flowvae_transport']['shifted_mu'] if 'shifted_mu' in train_config['flowvae_transport'] else 0.0
    assert shift_lg or ((not shifted_mu) and (shifted_mu == 0.0))
    transport = create_transport(
        train_config['flowvae_transport']['path_type'],
        train_config['flowvae_transport']['prediction'],
        train_config['flowvae_transport']['loss_weight'],
        train_config['flowvae_transport']['train_eps'],
        train_config['flowvae_transport']['sample_eps'],
        use_cosine_loss = train_config['flowvae_transport']['use_cosine_loss'] if 'use_cosine_loss' in train_config['flowvae_transport'] else False,
        use_lognorm = train_config['flowvae_transport']['use_lognorm'] if 'use_lognorm' in train_config['flowvae_transport'] else False,
        shift_lg = shift_lg, 
    )  # default: velocity;
    sampler = Sampler(transport)
    sample_fn = sampler.sample_ode(
        sampling_method=train_config['flowvae_sample']['sampling_method'],
        num_steps=train_config['flowvae_sample']['num_sampling_steps'] + 1,
        atol=train_config['flowvae_sample']['atol'],
        rtol=train_config['flowvae_sample']['rtol'],
        reverse=train_config['flowvae_sample']['reverse'],
        timestep_shift=timestep_shift,
    )
    
    using_cfg = cfg_scale > 1.0
    if using_cfg:
        if accelerator.process_index == 0:
            print_with_prefix('Using cfg:', using_cfg)

    if rank == 0:
        os.makedirs(sample_folder_dir, exist_ok=True)
        if accelerator.process_index == 0 and not demo_sample_mode:
            print_with_prefix(f"Saving .png samples at {sample_folder_dir}")
    accelerator.wait_for_everyone()
    
    # ---------get latent info-----------------------
    latent_norm = train_config['data']['latent_norm'] if 'latent_norm' in train_config['data'] else False
    latent_dataset = ImgLatentDataset(
        data_dir=train_config['data']['data_path'],
        latent_norm=latent_norm,
        latent_multiplier=train_config['data']['latent_multiplier'] if 'latent_multiplier' in train_config['data'] else 0.18215,
    )
    latent_multiplier = train_config['data']['latent_multiplier'] if 'latent_multiplier' in train_config['data'] else 0.18215
    # move to device
    if latent_norm:
        latent_mean, latent_std = latent_dataset.get_latent_stats()
    else:
        latent_mean, latent_std = torch.tensor(0.0), torch.tensor(1.0)
    latent_mean = latent_mean.clone().detach().to(device)
    latent_std = latent_std.clone().detach().to(device)
    # ---------------------------------------------------

    # Image preprocessing
    image_size = train_config['data']['image_size']
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    # Create dataset and dataloader
    dataset = ImageFolder(root=train_config['data']['raw_val_data_dir'], transform=transform)
    distributed_sampler = DistributedSampler(dataset, num_replicas=accelerator.num_processes, rank=rank)
    val_dataloader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=False,
        num_workers=16,
        sampler=distributed_sampler
    )

    if accelerator.process_index == 0:
        print_with_prefix(f"Latent mean: {latent_mean}")
        print_with_prefix(f"Latent std: {latent_std}")
        print_with_prefix(f"Latent multiplier: {latent_multiplier}")

    init_std = train_config['flowvae_transport']['init_std'] if 'init_std' in train_config['flowvae_transport'] else 1.0
    
    fid, avg_psnr, avg_lpips, avg_ssim = 0, 0, 0, 0
    if demo_sample_mode:
        if accelerator.process_index == 0:
            images = []
            for i in tqdm([2070, 3600, 3870, 9740, 880, 9790, 4170, 2790], desc="Generating Demo Samples"):
                img = dataset.__getitem__(i)[0].unsqueeze(0).to(device)
                y_ = model.encode(img).sample() if vae is None else vae.encode_images(img)
                y = (y_ - latent_mean) / latent_std * latent_multiplier
                z = torch.randn(1, 3, image_size, image_size, device=device) * init_std
                if using_cfg:
                    z = torch.cat([z, z], 0)
                    y_null = torch.zeros_like(y, device=device)
                    y = torch.cat([y, y_null], 0)
                    model_kwargs = dict(y=y, cfg_scale=cfg_scale, cfg_interval=False, cfg_interval_start=cfg_interval_start)
                    model_fn = model.forward_with_cfg
                else:
                    model_kwargs = dict(y=y)
                    model_fn = model.forward
                samples = sample_fn(z, model_fn, **model_kwargs)[-1]
                samples = torch.cat([samples[:1], img, model.decode(y_)], dim=0)
                samples = torch.clamp(127.5 * samples + 128.0, 0, 255).permute(0, 2, 3, 1).to("cpu", dtype=torch.uint8).numpy()
                images.append(samples)
            # Combine 8 images into a 6x4 grid
            # Stack all images into a large numpy array
            all_images = np.concatenate(images)  # Take first image from each batch            
            # Rearrange into 6x4 grid
            h, w = all_images.shape[1:3]
            grid = np.zeros((4 * h, 6 * w, 3), dtype=np.uint8)
            for idx, image in enumerate(all_images):
                i, j = divmod(idx, 6)  # Calculate position in 4x6 grid
                grid[i*h:(i+1)*h, j*w:(j+1)*w] = image
                
            # Save the combined image
            Image.fromarray(grid).save(f"{train_config['train']['output_dir']}/demo_samples_cfg{cfg_scale}.png")
        accelerator.wait_for_everyone()
    else:
        # Setup output directories
        save_dir = sample_folder_dir
        ref_path = os.path.join(train_config['train']['output_dir'], 'ref_images')
        os.makedirs(save_dir, exist_ok=True)
        os.makedirs(ref_path, exist_ok=True)

        local_rank = accelerator.process_index
        world_size = accelerator.num_processes
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
                        print_with_prefix(f"Saved {total_samples * world_size} reference images")
        accelerator.wait_for_everyone()

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
            
            with torch.no_grad():
                # Sample inputs:
                z = torch.randn(images.shape[0], 3, image_size, image_size, device=device) * init_std
                y_ = model.encode(images).sample() if vae is None else vae.encode_images(images)
                y = (y_ - latent_mean) / latent_std * latent_multiplier
                
                # Setup classifier-free guidance:
                if using_cfg:
                    z = torch.cat([z, z], 0)
                    y_null = torch.zeros_like(y, device=device)
                    y = torch.cat([y, y_null], 0)
                    model_kwargs = dict(y=y, cfg_scale=cfg_scale, cfg_interval=True, cfg_interval_start=cfg_interval_start)
                    model_fn = model.forward_with_cfg
                else:
                    model_kwargs = dict(y=y)
                    model_fn = model.forward
                samples = sample_fn(z, model_fn, **model_kwargs)[-1]
                if using_cfg:
                    samples, _ = samples.chunk(2, dim=0)  # Remove null class samples
                decoded_images_tensor = samples
                decoded_images = torch.clamp(127.5 * decoded_images_tensor + 128.0, 0, 255).permute(0, 2, 3, 1).to("cpu", dtype=torch.uint8).numpy()
            
            # Compute metrics
            lpips_values.append(lpips(decoded_images_tensor, images).mean())
            ssim_values.append(ssim_metric(decoded_images_tensor, images))
            
            # Save reconstructions
            for i, img in enumerate(decoded_images):
                save_image(img, os.path.join(save_dir, f"decoded_image_rank_{local_rank}_{all_indices + i}.png"))
                if (all_indices + i) % 100 == 0 and local_rank == 0 and i != 0:
                    print_with_prefix(f"Processed {(all_indices + i) * world_size} images")
            all_indices += len(decoded_images)
        accelerator.wait_for_everyone()

        # Aggregate metrics across GPUs
        lpips_values = torch.tensor(lpips_values).to(device)
        ssim_values = torch.tensor(ssim_values).to(device)
        dist.all_reduce(lpips_values, op=dist.ReduceOp.AVG)
        dist.all_reduce(ssim_values, op=dist.ReduceOp.AVG)
        
        avg_lpips = lpips_values.mean().item()
        avg_ssim = ssim_values.mean().item()

        fid, avg_psnr = 0, 0
        if local_rank == 0:
            # Calculate FID
            print_with_prefix("Computing rFID...")
            fid = calculate_fid_given_paths([ref_path, save_dir], batch_size=200, dims=2048, device=device, num_workers=32)

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
            with open(sample_folder_dir.replace('_decoded_images', '.txt'), 'w') as f:
                f.write(f"rFID: {fid:.3f}\nPSNR: {avg_psnr:.3f}\nLPIPS: {avg_lpips:.3f}\nSSIM: {avg_ssim:.3f}")
        accelerator.wait_for_everyone()

    return fid, avg_psnr, avg_lpips, avg_ssim

# some utils
def print_with_prefix(*messages):
    prefix = f"\033[34m[FlowVAE-Sampling {strftime('%Y-%m-%d %H:%M:%S')}]\033[0m"
    combined_message = ' '.join(map(str, messages))
    print(f"{prefix}: {combined_message}")

def load_config(config_path):
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config


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

    # read config
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/dit_b_ldmvae_f16d16.yaml')
    parser.add_argument('--demo', action='store_true', default=False)
    args = parser.parse_args()
    train_config = load_config(args.config)
    # mixed_precision = train_config['flowvae_sample']['precision'] if 'precision' in train_config['flowvae_transport'] else 'no'
    accelerator = Accelerator()

    # get ckpt_dir
    assert 'ckpt_path' in train_config, "ckpt_path must be specified in config"
    if accelerator.process_index == 0:
        print_with_prefix('Using ckpt:', train_config['ckpt_path'])
    ckpt_dir = train_config['ckpt_path']

    # get model
    kwargs = dict(
        input_size=train_config['data']['image_size'],
        norm_type=train_config['vae']['norm_type'] if 'norm_type' in train_config['vae'] else 'rmsnorm',
        multi_latent=train_config['vae']['multi_latent'] if 'multi_latent' in train_config['vae'] else True,
        add_y_to_x=train_config['vae']['add_y_to_x'] if 'add_y_to_x' in train_config['vae'] else False,
        ckpt_path=train_config['vae']['model_path'] if 'model_path' in train_config['vae'] else False,
        sample_mode=True, 
    )
    model = VAE_Models[train_config['vae']['vae_type']](**kwargs)

    checkpoint = torch.load(ckpt_dir, map_location=lambda storage, loc: storage)
    if "ema" in checkpoint:  # supports checkpoints from train.py
        checkpoint = checkpoint["ema"]
    # if "model" in checkpoint:  # supports checkpoints from train.py
    #     checkpoint = checkpoint["model"]
    #     checkpoint = {k.replace('module.', ''): v for k, v in checkpoint.items()}
    model.load_state_dict(checkpoint, strict=False)
    model.eval()  # important!
    model.to(accelerator.device)

    fid, avg_psnr, avg_lpips, avg_ssim = do_sample(train_config, accelerator, ckpt_path=ckpt_dir, model=model, demo_sample_mode=args.demo)
    
