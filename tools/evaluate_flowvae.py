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
from accelerate import Accelerator
from torchvision import transforms
from torch.utils.data import DataLoader, DistributedSampler
from concurrent.futures import ThreadPoolExecutor, as_completed
from torchmetrics import StructuralSimilarityIndexMeasure
from torchvision.datasets import ImageFolder
# local imports

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from diffusion import create_diffusion
from transport import create_transport, Sampler
from models import Models
from tokenizer import VAE_Models
from datasets.img_latent_dataset import ImgLatentDataset
from tools.save_npz import create_npz_from_sample_folder
    
# sample function
def do_sample(train_config, accelerator, ckpt_path=None, cfg_scale=None, model=None, demo_sample_mode=False):
    """
    Run sampling.
    """
    folder_name = f"{train_config['model']['model_type'].replace('/', '-')}-ckpt-{ckpt_path.split('/')[-1].split('.')[0]}-{train_config['sample']['num_sampling_steps']}".lower()

    if cfg_scale is None:
        cfg_scale = train_config['sample']['cfg_scale']
    cfg_interval_start = train_config['sample']['cfg_interval_start'] if 'cfg_interval_start' in train_config['sample'] else 0
    timestep_shift = train_config['sample']['timestep_shift'] if 'timestep_shift' in train_config['sample'] else 0
    if cfg_scale > 1.0:
        folder_name += f"-interval{cfg_interval_start:.2f}"+f"-cfg{cfg_scale:.2f}"
        folder_name += f"-shift{timestep_shift:.2f}"

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
    else:
        png_files = [f for f in os.listdir(sample_folder_dir) if f.endswith('.png')]
        png_count = len(png_files)
        if png_count > train_config['sample']['fid_num'] and not demo_sample_mode:
            if accelerator.process_index == 0:
                print_with_prefix(f"Found {png_count} PNG files in {sample_folder_dir}, skip sampling.")
            return sample_folder_dir

    torch.backends.cuda.matmul.allow_tf32 = True  # True: fast but may lead to some small numerical differences
    assert torch.cuda.is_available(), "Sampling with DDP requires at least one GPU. sample.py supports CPU-only usage"
    torch.set_grad_enabled(False)

    # Setup accelerator:
    device = accelerator.device

    # Setup DDP:
    device = accelerator.device
    seed = train_config['train']['global_seed'] * accelerator.num_processes + accelerator.process_index
    torch.manual_seed(seed)
    # torch.cuda.set_device(device)
    print_with_prefix(f"Starting rank={accelerator.local_process_index}, seed={seed}, world_size={accelerator.num_processes}.")
    rank = accelerator.local_process_index

    # Load model:
    if 'downsample_ratio' in train_config['vae']:
        downsample_ratio = train_config['vae']['downsample_ratio']
    else:
        downsample_ratio = 16
    latent_size = train_config['data']['image_size'] // downsample_ratio


    transport = create_transport(
        train_config['transport']['path_type'],
        train_config['transport']['prediction'],
        train_config['transport']['loss_weight'],
        train_config['transport']['train_eps'],
        train_config['transport']['sample_eps'],
        use_cosine_loss = train_config['transport']['use_cosine_loss'] if 'use_cosine_loss' in train_config['transport'] else False,
        use_lognorm = train_config['transport']['use_lognorm'] if 'use_lognorm' in train_config['transport'] else False,
    )  # default: velocity;
    sampler = Sampler(transport)
    sample_fn = sampler.sample_ode(
        sampling_method=train_config['sample']['sampling_method'],
        num_steps=train_config['sample']['num_sampling_steps'],
        atol=train_config['sample']['atol'],
        rtol=train_config['sample']['rtol'],
        reverse=train_config['sample']['reverse'],
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

    # Figure out how many samples we need to generate on each GPU and how many iterations we need to run:
    n = train_config['sample']['per_proc_batch_size']
    global_batch_size = n * accelerator.num_processes
    # To make things evenly-divisible, we'll sample a bit more than we need and then discard the extra samples:
    num_samples = len([name for name in os.listdir(sample_folder_dir) if (os.path.isfile(os.path.join(sample_folder_dir, name)) and ".png" in name)])
    total_samples = int(math.ceil(train_config['sample']['fid_num'] / global_batch_size) * global_batch_size)
    if not demo_sample_mode and rank == 0:
        if accelerator.process_index == 0:
            print_with_prefix(f"Total number of images that will be sampled: {total_samples}")
    assert total_samples % accelerator.num_processes == 0, "total_samples must be divisible by world_size"
    samples_needed_this_gpu = int(total_samples // accelerator.num_processes)
    assert samples_needed_this_gpu % n == 0, "samples_needed_this_gpu must be divisible by the per-GPU batch size"
    iterations = int(samples_needed_this_gpu // n)
    done_iterations = int( int(num_samples // accelerator.num_processes) // n)
    pbar = range(iterations)
    if not demo_sample_mode:
        pbar = tqdm(pbar) if rank == 0 else pbar
    total = 0
    
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
    dataset = ImageFolder(root=train_config['data']['raw_data_dir'], transform=transform)
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

    init_std = train_config['transport']['init_std'] if 'init_std' in train_config['transport'] else 1.0
    if demo_sample_mode:
        if accelerator.process_index == 0:
            images = []
            for i in tqdm([2070, 3600, 3870, 9740, 880, 9790, 4170, 2790], desc="Generating Demo Samples"):
                img = dataset.__getitem__(i)[0].unsqueeze(0).to(device)
                y_ = model.encode(img).sample()
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

            return None
    else:
        for i in pbar:
            # Sample inputs:
            z = torch.randn(n, model.in_channels, latent_size, latent_size, device=device)
            y = torch.randint(0, train_config['data']['num_classes'], (n,), device=device)
            
            # Setup classifier-free guidance:
            if using_cfg:
                z = torch.cat([z, z], 0)
                y_null = torch.tensor([1000] * n, device=device)
                y = torch.cat([y, y_null], 0)
                model_kwargs = dict(y=y, cfg_scale=cfg_scale, cfg_interval=True, cfg_interval_start=cfg_interval_start)
                model_fn = model.forward_with_cfg
            else:
                model_kwargs = dict(y=y)
                model_fn = model.forward

            samples = sample_fn(z, model_fn, **model_kwargs)[-1]
            if using_cfg:
                samples, _ = samples.chunk(2, dim=0)  # Remove null class samples

            samples = (samples * latent_std) / latent_multiplier + latent_mean
            samples = vae.decode_to_images(samples)

            # Save samples to disk as individual .png files
            for i, sample in enumerate(samples):
                index = i * accelerator.num_processes + accelerator.process_index + total
                Image.fromarray(sample).save(f"{sample_folder_dir}/{index:06d}.png")
            total += global_batch_size
            accelerator.wait_for_everyone()
        if accelerator.process_index == 0:
            create_npz_from_sample_folder(sample_folder_dir, train_config['sample']['fid_num'])

    return sample_folder_dir

# some utils
def print_with_prefix(*messages):
    prefix = f"\033[34m[DiT-Sampling {strftime('%Y-%m-%d %H:%M:%S')}]\033[0m"
    combined_message = ' '.join(map(str, messages))
    print(f"{prefix}: {combined_message}")

def load_config(config_path):
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config

if __name__ == "__main__":

    # read config
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/dit_b_ldmvae_f16d16.yaml')
    parser.add_argument('--demo', action='store_true', default=False)
    args = parser.parse_args()
    train_config = load_config(args.config)
    # mixed_precision = train_config['sample']['precision'] if 'precision' in train_config['transport'] else 'no'
    accelerator = Accelerator()

    # get ckpt_dir
    assert 'ckpt_path' in train_config, "ckpt_path must be specified in config"
    if accelerator.process_index == 0:
        print_with_prefix('Using ckpt:', train_config['ckpt_path'])
    ckpt_dir = train_config['ckpt_path']

    if 'downsample_ratio' in train_config['vae']:
        latent_size = train_config['data']['image_size'] // train_config['vae']['downsample_ratio']
    else:
        latent_size = train_config['data']['image_size'] // 8

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
    # if "ema" in checkpoint:  # supports checkpoints from train.py
    #     checkpoint = checkpoint["ema"]
    if "model" in checkpoint:  # supports checkpoints from train.py
        checkpoint = checkpoint["model"]
        checkpoint = {k.replace('module.', ''): v for k, v in checkpoint.items()}
    model.load_state_dict(checkpoint, strict=False)
    model.eval()  # important!
    model.to(accelerator.device)

    # naive sample
    sample_folder_dir = do_sample(train_config, accelerator, ckpt_path=ckpt_dir, model=model, demo_sample_mode=args.demo)
    
    if not args.demo:
        # calculate FID
        # Important: FID is only for reference, please use ADM evaluation for paper reporting
        if accelerator.process_index == 0:
            from tools.calculate_fid import calculate_fid_given_paths
            print_with_prefix('Calculating FID with {} number of samples'.format(train_config['sample']['fid_num']))
            assert 'fid_reference_file' in train_config['data'], "fid_reference_file must be specified in config"
            fid_reference_file = train_config['data']['fid_reference_file']
            fid = calculate_fid_given_paths(
                [fid_reference_file, sample_folder_dir],
                batch_size=200,
                dims=2048,
                device='cuda',
                num_workers=16,
                sp_len = train_config['sample']['fid_num']
            )
            print_with_prefix('fid=',fid)
