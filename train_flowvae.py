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
from transport import create_transport
from tokenizer import VAE_Models
from accelerate import Accelerator
from datasets.img_latent_dataset import ImgLatentDataset

def load_weights_with_shape_check(model, checkpoint, rank=0):
    model_state_dict = model.state_dict()
    # check shape and load weights
    checkpoint = checkpoint['model'] if 'model' in checkpoint else checkpoint
    for name, param in checkpoint.items():
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

    kwargs = dict(
        input_size=train_config['data']['image_size'],
        norm_type=train_config['vae']['norm_type'] if 'norm_type' in train_config['vae'] else 'rmsnorm',
        multi_latent=train_config['vae']['multi_latent'] if 'multi_latent' in train_config['vae'] else True,
        add_y_to_x=train_config['vae']['add_y_to_x'] if 'add_y_to_x' in train_config['vae'] else False,
        ckpt_path=train_config['vae']['model_path'] if 'model_path' in train_config['vae'] else False,
        upsampler=train_config['vae']['upsampler'] if 'upsampler' in train_config['vae'] else 'nearest',
        sample_mode=False, 
    )
    model = VAE_Models[train_config['vae']['vae_type']](**kwargs)

    ema = deepcopy(model).to(device)  # Create an EMA of the model for use after training
    # load pretrained model
    if 'weight_init' in train_config['train']:
        if model.model_type == 'vavae':
            sd = torch.load(train_config['train']['weight_init'], map_location="cpu")
            if 'state_dict' in sd.keys():
                sd = sd['state_dict']
            sd = {k: v for k, v in sd.items() if 'foundation_model.model' not in k and 'loss' not in k}
        elif model.model_type == 'sdvae':
            from safetensors.torch import load_file
            sd = load_file(train_config['train']['weight_init'])
        elif model.model_type == 'marvae':
            sd = torch.load(train_config['train']['weight_init'], map_location="cpu")["model"]
        else:
            raise ValueError(f"Invalid model type: {model.model_type}")

        model = load_weights_with_shape_check(model, sd, rank=rank)
        ema = load_weights_with_shape_check(ema, sd, rank=rank)
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
        logger.info(f"FlowVAE Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
        logger.info(f"FlowVAE Trainable Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.2f}M")
        logger.info(f"Optimizer: AdamW, lr={train_config['optimizer']['lr']}, beta2={train_config['optimizer']['beta2']}")
        for name, param in model.named_parameters():
            logger.info(f"{name+'.requires_grad':<60}: {param.requires_grad}")

    opt = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), 
        lr=train_config['optimizer']['lr'], 
        weight_decay=0, 
        betas=(0.9, train_config['optimizer']['beta2'])
        )
    
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
        raw_img_transform=raw_img_transform if raw_data_dir is not None else None, 
        raw_img_drop=train_config['data']['raw_img_drop'] if 'raw_img_drop' in train_config['data'] else 0.0,
    )    
    else:
        from tools.extract_features import center_crop_arr
        from torchvision import transforms
        from torchvision.datasets import ImageFolder
        transform = transforms.Compose([
            transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, crop_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
        ])
        dataset = ImageFolder(train_config['data']['data_path'], transform=transform)

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
            checkpoint_files.sort(key=lambda x: os.path.getsize(x))
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
    running_grad_norm = 0
    eval_every = train_config['train']['eval_every'] if 'eval_every' in train_config['train'] else math.inf
    start_time = time()
    if accelerator.is_main_process:
        logger.info(f"Train config: {train_config}")

    while True:
        for latent, label, raw_img in loader:
            x = raw_img.to(device, non_blocking=True)
            y = latent.to(device, non_blocking=True)  
            model_kwargs = dict(y=y)
            loss_dict = transport.training_losses(
                model, x, model_kwargs, 
                l2_loss=train_config['transport']['l2_loss'] if 'l2_loss' in train_config['transport'] else True
                )
            
            if 'cos_loss' in loss_dict:
                mse_loss = loss_dict["loss"].mean()
                loss = loss_dict["cos_loss"].mean() + mse_loss
            else:
                loss = loss_dict["loss"].mean()
            opt.zero_grad()
            accelerator.backward(loss)
            if 'max_grad_norm' in train_config['optimizer']:
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), train_config['optimizer']['max_grad_norm'])
            running_grad_norm += get_grad_norm(model)
            opt.step()
            update_ema(ema, model.module)

            # Log loss values:
            if 'cos_loss' in loss_dict:
                running_loss += mse_loss.item()
            else:
                running_loss += loss.item()
            log_steps += 1
            train_steps += 1
            if train_steps % train_config['train']['log_every'] == 0:
                # Measure training speed:
                torch.cuda.synchronize()
                end_time = time()
                steps_per_sec = log_steps / (end_time - start_time)
                # Reduce loss history over all processes:
                avg_loss = torch.tensor(running_loss / log_steps, device=device)
                dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
                avg_loss = avg_loss.item() / dist.get_world_size()
                # Reduce loss history over all processes:
                avg_grad_norm = torch.tensor(running_grad_norm / log_steps, device=device)
                dist.all_reduce(avg_grad_norm, op=dist.ReduceOp.SUM)
                avg_grad_norm = avg_grad_norm.item() / dist.get_world_size()
                if accelerator.is_main_process:
                    logger.info(f"(step={train_steps:07d}) Train Loss: {avg_loss:.4f}, Train Steps/Sec: {steps_per_sec:.2f}, Grad Norm: {avg_grad_norm:.4f}")
                    accelerator.log({"loss": avg_loss, "grad_norm": avg_grad_norm}, step=train_steps)
                # Reset monitoring variables:
                running_loss = 0
                running_grad_norm = 0
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
            # if (train_steps % eval_every == 0 and train_steps > 0) or train_steps == train_config['train']['max_steps']:
            if (train_steps % eval_every == 0 and train_steps > 0):
                from inference import do_sample

                # stored
                temp_stored_params = [param.detach().cpu().clone() for param in model.parameters()]
                # copy ema to model
                for s_param, param in zip(ema.parameters(), model.parameters()):
                    param.data.copy_(s_param.to(param.device).data)
                # sampling without cfg
                model.eval()
                temp_cfg = train_config['sample']['cfg_scale']
                train_config['sample']['cfg_scale'] = 1.0
                with torch.no_grad():
                    sample_folder_dir = do_sample(
                        train_config, accelerator, ckpt_path=f"{train_steps:07d}.pt", model=model.module.module, vae=vae)
                train_config['sample']['cfg_scale'] = temp_cfg
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
    torch.backends.cudnn.benchmark = True

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