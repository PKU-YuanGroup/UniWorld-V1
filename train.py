# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DiT: https://github.com/facebookresearch/DiT
# LightningDiT: https://github.com/hustvl/LightningDiT
# --------------------------------------------------------
import random
import torch
# the first flag below was False when we tested this script but True makes A100 training a lot faste
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import torch.distributed as dist
import torch.backends.cuda
import torch.backends.cudnn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from accelerate.utils import ProjectConfiguration
import yaml
import numpy as np
import logging
import os
import argparse
from time import time
import math
from glob import glob
from copy import deepcopy
from collections import OrderedDict
# local imports
from diffusion import create_diffusion
from models import Models
from tokenizer import VAE_Models
from transport import create_transport
from accelerate import Accelerator
from datasets.img_latent_dataset import ImgLatentDataset

def load_weights_with_shape_check(model, checkpoint, rank=0):
    model_state_dict = model.state_dict()
    # check shape and load weights
    for name, param in checkpoint['model'].items():
        if name in model_state_dict:
            if param.shape == model_state_dict[name].shape:
                model_state_dict[name].copy_(param)
            else:
                if rank == 0:
                    print(f"Skipping loading parameter '{name}' due to shape mismatch: "
                        f"checkpoint shape {param.shape}, model shape {model_state_dict[name].shape}")
        else:
            if rank == 0:
                print(f"Parameter '{name}' not found in model, skipping.")
    # load state dict
    model.load_state_dict(model_state_dict, strict=False)
    
    return model

def get_grad_norm(model, norm_type=2):
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(norm_type)
            total_norm += param_norm.item() ** norm_type
    return total_norm ** (1.0 / norm_type)

def reduce_over_all_processes(running_data, log_steps, device):
    avg_data = torch.tensor(running_data / log_steps, device=device)
    dist.all_reduce(avg_data, op=dist.ReduceOp.SUM)
    avg_data = avg_data.item() / dist.get_world_size()
    return avg_data

def do_train(train_config, accelerator):
    """
    Trains a DiT.
    """
    # Setup accelerator:
    device = accelerator.device

    # Setup an experiment folder:
    if accelerator.is_main_process:
        logger = create_logger(train_config['train']['output_dir'])
        
    checkpoint_dir = f"{train_config['train']['output_dir']}/checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)  # Make results folder (holds all experiment subfolders)
        

    # get rank
    rank = accelerator.local_process_index 

    # Create model:
    offline_features = train_config['data']['offline_features'] if 'offline_features' in train_config['data'] else True
    vae = None if offline_features else VAE_Models[train_config['vae']['vae_type']](train_config['vae']['model_path']).eval()

    if 'downsample_ratio' in train_config['vae']:
        downsample_ratio = train_config['vae']['downsample_ratio']
    else:
        downsample_ratio = 8
    assert train_config['data']['image_size'] % downsample_ratio == 0, "Image size must be divisible by 8 (for the VAE encoder)."
    latent_size = train_config['data']['image_size'] // downsample_ratio
    use_diffusion = train_config['scheduler']['diffusion'] if 'diffusion' in train_config['scheduler'] else False
    use_transport = train_config['scheduler']['transport'] if 'transport' in train_config['scheduler'] else False
    assert (use_diffusion ^ use_transport), "use_diffusion and use_transport must be different (one True, one False)"
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
        num_timestep_token=train_config['model']['num_timestep_token'] if 'num_timestep_token' in train_config['model'] else 1,
        num_label_token=train_config['model']['num_label_token'] if 'num_label_token' in train_config['model'] else 1,
    )
    if 'without_timestep' in train_config['model']:
        kwargs.update(dict(without_timestep=train_config['model']['without_timestep']))
    model = Models[train_config['model']['model_type']](**kwargs)

    ema = deepcopy(model).to(device)  # Create an EMA of the model for use after training
    # load pretrained model
    if 'weight_init' in train_config['train']:
        checkpoint = torch.load(train_config['train']['weight_init'], map_location=lambda storage, loc: storage)
        # remove the prefix 'module.' from the keys
        checkpoint['model'] = {k.replace('module.', ''): v for k, v in checkpoint['model'].items()}
        model = load_weights_with_shape_check(model, checkpoint, rank=rank)
        ema = load_weights_with_shape_check(ema, checkpoint, rank=rank)
        if accelerator.is_main_process:
            logger.info(f"Loaded pretrained model from {train_config['train']['weight_init']}")

    requires_grad(ema, False)
    train_module = train_config['train']['train_module'] if 'train_module' in train_config['train'] else False
    if train_module and not ('all' in train_module):
        model.requires_grad_(False)
        for name, param in model.named_parameters():
            for n in train_module:
                if n in name:
                    param.requires_grad = True
                    break
    
    model = DDP(model.to(device), device_ids=[device])

    if use_diffusion:
        diffusion = create_diffusion(
            timestep_respacing="", 
            learn_sigma=train_config['diffusion']['learn_sigma'], 
            diffusion_steps=train_config['diffusion']['diffusion_steps'], 
            )  # default: 1000 steps, linear noise schedule
    else:    
        timestep_sampling = train_config['transport']['timestep_sampling'] if 'timestep_sampling' in train_config['transport'] else 'lognorm'
        shift_lg = train_config['transport']['shift_lg'] if 'shift_lg' in train_config['transport'] else True
        shifted_mu = train_config['transport']['shifted_mu'] if 'shifted_mu' in train_config['transport'] else 0.0
        assert (timestep_sampling == 'lognorm') or ((timestep_sampling != 'lognorm') and (not shift_lg))
        assert shift_lg or ((not shifted_mu) and (shifted_mu == 0.0))
        transport = create_transport(
            train_config['transport']['path_type'],
            train_config['transport']['prediction'],
            train_config['transport']['loss_weight'],
            train_config['transport']['train_eps'],
            train_config['transport']['sample_eps'],
            use_cosine_loss = train_config['transport']['use_cosine_loss'] if 'use_cosine_loss' in train_config['transport'] else False,
            use_lognorm = train_config['transport']['use_lognorm'] if 'use_lognorm' in train_config['transport'] else False,
        )  # default: velocity; 

    if accelerator.is_main_process:
        logger.info(f"Model: {model}")
        logger.info(f"DiT Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
        logger.info(f"DiT Trainable Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.2f}M")
        logger.info(f"Optimizer: AdamW, lr={train_config['optimizer']['lr']}, beta2={train_config['optimizer']['beta2']}")
        for name, param in model.named_parameters():
            logger.info(f"{name+'.requires_grad':<60}: {param.requires_grad}")

    optimizer_name = train_config['optimizer']['optimizer_name'] if 'optimizer_name' in train_config['optimizer'] else 'adamw'
    if optimizer_name == 'adamw':
        opt = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()), 
            lr=train_config['optimizer']['lr'], 
            weight_decay=0, 
            betas=(0.9, train_config['optimizer']['beta2'])
            )
    elif optimizer_name == 'muon':
        from tools.muon import Muon
        muon_params = [p for name, p in model.named_parameters() if p.ndim == 2 and p.requires_grad]
        adamw_params = [p for name, p in model.named_parameters() if p.ndim != 2 and p.requires_grad]
        opt = Muon(
            lr=train_config['optimizer']['lr'],
            wd=0.0,
            muon_params=muon_params,
            adamw_params=adamw_params,
        )
    else:
        assert 0, "optimizer not supported"
    

    # Setup data
    uncondition = train_config['data']['uncondition'] if 'uncondition' in train_config['data'] else False
    raw_data_dir = train_config['data']['raw_data_dir'] if 'raw_data_dir' in train_config['data'] else None
    crop_size = train_config['data']['image_size']
    if offline_features:
        from tools.extract_features import center_crop_arr
        from torchvision import transforms
        raw_img_transform = transforms.Compose([
            transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, crop_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
        ])
        dataset = ImgLatentDataset(
            data_dir=train_config['data']['data_path'],
            latent_norm=train_config['data']['latent_norm'] if 'latent_norm' in train_config['data'] else False,
            latent_multiplier=train_config['data']['latent_multiplier'], 
            raw_data_dir=raw_data_dir, 
            raw_img_transform=raw_img_transform if raw_data_dir is not None else None
        )    
    else:
        from tools.extract_features import center_crop_arr, random_crop_arr
        from torchvision import transforms
        from torchvision.datasets import ImageFolder

        transform = transforms.Compose([
            transforms.Lambda(lambda pil_image: random_crop_arr(pil_image, crop_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
        ])
        dataset = ImageFolder(train_config['data']['raw_data_dir'], transform=transform)

    batch_size_per_gpu = int(np.round(train_config['train']['global_batch_size'] / accelerator.num_processes))
    global_batch_size = batch_size_per_gpu * accelerator.num_processes
    loader = DataLoader(
        dataset,
        batch_size=batch_size_per_gpu,
        shuffle=True,
        num_workers=train_config['data']['num_workers'],
        pin_memory=True,
        drop_last=True
    )
    if accelerator.is_main_process:
        logger.info(f"Dataset contains {len(dataset):,} images {train_config['data']['data_path']}")
        logger.info(f"Batch size {batch_size_per_gpu} per gpu, with {global_batch_size} global batch size")
    
    # Prepare models for training:
    update_ema(ema, model.module, decay=0)  # Ensure EMA is initialized with synced weights
    model.train()  # important! This enables embedding dropout for classifier-free guidance
    ema.eval()  # EMA model should always be in eval mode
    
    train_config['train']['resume'] = train_config['train']['resume'] if 'resume' in train_config['train'] else False

    if train_config['train']['resume']:
        # check if the checkpoint exists
        checkpoint_files = glob(f"{checkpoint_dir}/*.pt")
        if checkpoint_files:
            checkpoint_files.sort(key=lambda x: int(x.split('/')[-1].split('.')[0]))
            latest_checkpoint = checkpoint_files[-1]
            checkpoint = torch.load(latest_checkpoint, map_location=lambda storage, loc: storage)
            model.load_state_dict(checkpoint['model'])
            opt.load_state_dict(checkpoint['opt'])
            ema.load_state_dict(checkpoint['ema'])
            train_steps = int(latest_checkpoint.split('/')[-1].split('.')[0])
            if accelerator.is_main_process:
                logger.info(f"Resuming training from checkpoint: {latest_checkpoint}")
        else:
            if accelerator.is_main_process:
                logger.info("No checkpoint found. Starting training from scratch.")
    model, opt, loader = accelerator.prepare(model, opt, loader)

    # Variables for monitoring/logging purposes:
    if not train_config['train']['resume']:
        train_steps = 0
    log_steps = 0
    running_loss = 0
    running_cos_loss = 0
    running_grad_norm = 0
    eval_every = train_config['train']['eval_every'] if 'eval_every' in train_config['train'] else math.inf
    start_time = time()
    if accelerator.is_main_process:
        logger.info(f"Train config: {train_config}")

    while True:
        for x, y, raw_img in loader:
            x = x.to(device, non_blocking=True)
            if uncondition:
                y = (torch.ones_like(y) * train_config['data']['num_classes']).to(y.dtype)
            y = y.to(device, non_blocking=True)  
            model_kwargs = dict(y=y)
            if use_diffusion:
                t = torch.randint(0, diffusion.num_timesteps, (x.shape[0],), device=device)
                loss_dict = diffusion.training_losses(model, x, t, model_kwargs)
                mse_loss = loss_dict["loss"].mean()
                cos_loss = torch.tensor(0.0)
            else:
                loss_dict = transport.training_losses(
                    model, x, model_kwargs, 
                    shifted_mu=shifted_mu, timestep_sampling=timestep_sampling
                    )
                mse_loss = loss_dict["loss"].mean()
                cos_loss = loss_dict["cos_loss"].mean() if 'cos_loss' in loss_dict else torch.tensor(0.0)
            loss = cos_loss + mse_loss

            opt.zero_grad()
            accelerator.backward(loss)
            if 'max_grad_norm' in train_config['optimizer']:
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), train_config['optimizer']['max_grad_norm'])
            running_grad_norm += get_grad_norm(model)
            opt.step()
            update_ema(ema, model.module)

            # Log loss values:
            running_loss += mse_loss.item()
            running_cos_loss += cos_loss.item()
            log_steps += 1
            train_steps += 1
            if train_steps % train_config['train']['log_every'] == 0:
                # Measure training speed:
                torch.cuda.synchronize()
                end_time = time()
                steps_per_sec = log_steps / (end_time - start_time)
                # Reduce loss history over all processes:
                avg_loss = reduce_over_all_processes(running_loss, log_steps, device=device)
                avg_grad_norm = reduce_over_all_processes(running_grad_norm, log_steps, device=device)
                avg_cos_loss = reduce_over_all_processes(running_cos_loss, log_steps, device=device)
                if accelerator.is_main_process:
                    info = f"(step={train_steps:07d}) Train Loss: {avg_loss:.4f}, Train Steps/Sec: {steps_per_sec:.2f}, Grad Norm: {avg_grad_norm:.4f}"
                    if avg_cos_loss != 0: info += f", Cos Loss: {avg_cos_loss:.4f}"
                    logger.info(info)

                    log_dict = dict(loss=avg_loss, grad_norm=avg_grad_norm)
                    if avg_cos_loss != 0: log_dict['cos_loss'] = avg_cos_loss
                    accelerator.log(log_dict, step=train_steps)

                # Reset monitoring variables:
                running_loss = 0
                running_grad_norm = 0
                running_cos_loss = 0
                log_steps = 0
                start_time = time()

            # Save checkpoint:
            if train_steps % train_config['train']['ckpt_every'] == 0 and train_steps > 0:
                if accelerator.is_main_process:
                    checkpoint = {
                        "model": model.module.state_dict(),
                        "ema": ema.state_dict(),
                        "opt": opt.state_dict(),
                        "config": train_config,
                    }
                    checkpoint_path = f"{checkpoint_dir}/{train_steps:07d}.pt"
                    torch.save(checkpoint, checkpoint_path)
                    if accelerator.is_main_process:
                        logger.info(f"Saved checkpoint to {checkpoint_path}")
                dist.barrier()

            # Eval online or last step:
            if (train_steps % eval_every == 0 and train_steps > 0) or train_steps == train_config['train']['max_steps']:
                from inference import do_sample

                # stored
                temp_stored_params = [param.detach().cpu().clone() for param in model.parameters()]
                # copy ema to model
                for s_param, param in zip(ema.parameters(), model.parameters()):
                    param.data.copy_(s_param.to(param.device).data)
                # sampling without cfg
                model.eval()
                temp_cfg, temp_steps = train_config['sample']['cfg_scale'], train_config['sample']['num_sampling_steps']
                train_config['sample']['cfg_scale'], train_config['sample']['num_sampling_steps'] = 1.0, 250
                with torch.no_grad():
                    sample_folder_dir = do_sample(
                        train_config, accelerator, ckpt_path=f"{train_steps:07d}.pt", model=model.module.module, vae=vae)
                    torch.cuda.empty_cache()
                train_config['sample']['cfg_scale'], train_config['sample']['num_sampling_steps'] = temp_cfg, temp_steps
                model.train()
                # restored
                for c_param, param in zip(temp_stored_params, model.parameters()):
                    param.data.copy_(c_param.data)
                temp_stored_params = None

                # calculate FID
                # Important: FID is only for reference, please use ADM evaluation for paper reporting
                if accelerator.process_index == 0:
                    from tools.calculate_fid import calculate_fid_given_paths
                    logger.info(f"Calculating FID with {train_config['sample']['fid_num']} number of samples")
                    assert 'fid_reference_file' in train_config['data'], "fid_reference_file must be specified in config"
                    fid_reference_file = train_config['data']['fid_reference_file']
                    fid = calculate_fid_given_paths(
                        [fid_reference_file, sample_folder_dir],
                        batch_size=200,
                        dims=2048,
                        device='cuda',
                        num_workers=16,
                        sp_len=train_config['sample']['fid_num']
                    )
                    logger.info(f"(step={train_steps:07d}), Fid={fid}")
                    accelerator.log({"fid": fid}, step=train_steps)

            if train_steps >= train_config['train']['max_steps']:
                break
        if train_steps >= train_config['train']['max_steps']:
            break

    if accelerator.is_main_process:
        logger.info("Done!")

    return accelerator


@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """
    Step the EMA model towards the current model.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        name = name.replace("module.", "")
        # TODO: Consider applying only to params that require_grad to avoid small numerical changes of pos_embed
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag

def load_config(config_path):
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config

def create_logger(logging_dir):
    """
    Create a logger that writes to a log file and stdout.
    """
    if dist.get_rank() == 0:  # real logger
        logging.basicConfig(
            level=logging.INFO,
            format='[\033[34m%(asctime)s\033[0m] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            handlers=[logging.StreamHandler(), logging.FileHandler(f"{logging_dir}/log.txt")]
        )
        logger = logging.getLogger(__name__)
    else:  # dummy logger (does nothing)
        logger = logging.getLogger(__name__)
        logger.addHandler(logging.NullHandler())
    return logger

def set_seed(seed, rank, device_specific=True):
    if device_specific:
        seed += rank
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

if __name__ == "__main__":
    # read config
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/debug.yaml')
    args = parser.parse_args()

    train_config = load_config(args.config)
    accelerator = Accelerator(
        mixed_precision=train_config['train']['precision'],
        log_with='wandb' if train_config['train']['wandb'] else None,
        project_config=ProjectConfiguration(project_dir=train_config['train']['output_dir']),
    )
    if train_config['train']['wandb'] and accelerator.is_main_process:
        import wandb
        wandb.login(key=train_config['wandb']['key'])
        wandb_init_kwargs = {"wandb": {"name": train_config['wandb']['log_name']}}
        accelerator.init_trackers(
            os.path.basename(train_config['wandb']['proj_name']), 
            config=train_config, init_kwargs=wandb_init_kwargs
            )
    
    set_seed(train_config['train']['seed'], accelerator.process_index, device_specific=True)
    do_train(train_config, accelerator)