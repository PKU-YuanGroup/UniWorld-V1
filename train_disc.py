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
from models import Models, LatentDiscriminator
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

def set_modules_requires_grad(modules, requires_grad):
    for module in modules:
        module.requires_grad_(requires_grad)
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
    # assert (use_diffusion ^ use_transport), "use_diffusion and use_transport must be different (one True, one False)"
    assert not use_diffusion
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
        class_dropout_prob=0.0, # do it after dataloader but before model.forward
    )
    model = Models[train_config['model']['model_type']](**kwargs)

    disc_config = train_config['discriminator']
    disc = LatentDiscriminator(
        disc_start=disc_config['disc_start'],
        disc_model=disc_config['disc_model'],
        disc_weight=disc_config['disc_weight'],
        disc_factor=disc_config['disc_factor'] if 'disc_factor' in disc_config else 1.0,
        complex_model=deepcopy(model) if disc_config['disc_model']['complex'] else None
    )

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
    disc = DDP(disc.to(device), device_ids=[device])

    if use_diffusion:
        diffusion = create_diffusion(
            timestep_respacing="", 
            learn_sigma=train_config['diffusion']['learn_sigma'], 
            diffusion_steps=train_config['diffusion']['diffusion_steps'], 
            )  # default: 1000 steps, linear noise schedule
    else:
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
        logger.info(f"Discriminator: {disc}")
        logger.info(f"DiT Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
        logger.info(f"DiT Trainable Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.2f}M")
        logger.info(f"Discriminator Parameters: {sum(p.numel() for p in disc.parameters()) / 1e6:.2f}M")
        logger.info(f"Discriminator Trainable Parameters: {sum(p.numel() for p in disc.parameters() if p.requires_grad) / 1e6:.2f}M")
        logger.info(f"Optimizer: AdamW, lr={train_config['optimizer']['lr']}, beta2={train_config['optimizer']['beta2']}")
        logger.info(f"Disc Optimizer: AdamW, lr={train_config['disc_optimizer']['lr']}, beta2={train_config['disc_optimizer']['beta2']}, weight_decay={train_config['disc_optimizer']['weight_decay']}")
        for name, param in model.named_parameters():
            logger.info(f"{name+'.requires_grad':<60}: {param.requires_grad}")
        for name, param in disc.named_parameters():
            logger.info(f"{name+'.requires_grad':<60}: {param.requires_grad}")

    modules_to_train = [module for module in model.module.get_trainable_modules()]
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

    disc_opt = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, disc.parameters()), 
        lr=train_config['disc_optimizer']['lr'], 
        weight_decay=train_config['disc_optimizer']['weight_decay'], 
        betas=(0.9, train_config['disc_optimizer']['beta2'])
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
        resume_disc = train_config['train']['resume_disc'] if 'resume_disc' in train_config['train'] else True
        # check if the checkpoint exists
        checkpoint_files = glob(f"{checkpoint_dir}/*.pt")
        if checkpoint_files:
            checkpoint_files.sort(key=lambda x: os.path.getsize(x))
            latest_checkpoint = checkpoint_files[-1]
            checkpoint = torch.load(latest_checkpoint, map_location=lambda storage, loc: storage)
            model.load_state_dict(checkpoint['model'])
            opt.load_state_dict(checkpoint['opt'])
            ema.load_state_dict(checkpoint['ema'])
            if resume_disc:
                disc.load_state_dict(checkpoint['disc'])
                disc_opt.load_state_dict(checkpoint['disc_opt'])
            train_steps = int(latest_checkpoint.split('/')[-1].split('.')[0])
            if accelerator.is_main_process:
                logger.info(f"Resuming training from checkpoint: {latest_checkpoint}")
        else:
            if accelerator.is_main_process:
                logger.info("No checkpoint found. Starting training from scratch.")
    
    model, opt, disc_opt, loader = accelerator.prepare(model, opt, disc_opt, loader)

    # Variables for monitoring/logging purposes:
    if not train_config['train']['resume']:
        train_steps = 0
    log_steps = 0
    running_loss = 0
    running_grad_norm = 0
    running_disc_grad_norm = 0
    running_g_loss = 0
    running_d_weight = 0
    running_gen_step_disc_factor = 0
    running_disc_loss = 0
    running_logits_real = 0
    running_logits_fake = 0
    running_disc_step_disc_factor = 0

    eval_every = train_config['train']['eval_every'] if 'eval_every' in train_config['train'] else math.inf
    delta_t_for_gt = train_config['discriminator']['delta_t_for_gt']
    delta_t_for_pred = train_config['discriminator']['delta_t_for_pred'] if 'delta_t_for_pred' in train_config['discriminator'] else delta_t_for_gt
    start_time = time()
    if accelerator.is_main_process:
        logger.info(f"Train config: {train_config}")

    while True:
        for x, y, raw_img in loader:
            x = x.to(device, non_blocking=True)
            if uncondition:
                y = (torch.ones_like(y) * train_config['data']['num_classes']).to(y.dtype)
            y = y.to(device, non_blocking=True)  

            drop_ids = torch.rand(y.shape[0], device=y.device) < 0.1 # 0.1 cfg
            y = torch.where(drop_ids, train_config['data']['num_classes'], y).contiguous()
            model_kwargs = dict(y=y)

            # select generator or discriminator
            if train_steps % 2 == 1 and train_steps >= disc.module.discriminator_iter_start:
                set_modules_requires_grad(modules_to_train, False)
                step_gen = False
                step_dis = True
            else:
                set_modules_requires_grad(modules_to_train, True)
                step_gen = True
                step_dis = False
                
            assert step_gen or step_dis, "You should backward either Gen. or Dis. in a step."

            if use_diffusion:
                t = torch.randint(0, diffusion.num_timesteps, (x.shape[0],), device=device)
                loss_dict = diffusion.training_losses(model, x, t, model_kwargs)
                cos_loss = torch.tensor(0.0, device=device)
                mse_loss = loss_dict["loss"].mean()
            else:
                loss_dict = transport.training_losses(model, x, model_kwargs)
                cos_loss = loss_dict["cos_loss"].mean() if 'cos_loss' in loss_dict else torch.tensor(0.0, device=device)
                mse_loss = loss_dict["loss"].mean()
            main_loss = cos_loss + mse_loss

            running_loss += mse_loss.item()
            ut = loss_dict['ut']
            xt = loss_dict['xt']
            pred = loss_dict['pred']
            t = loss_dict['t'][:, None, None, None]
            if step_gen:
                # ut = x1 - x0
                # xt = t * x1 + (1-t) * x0
                # x1 = xt + (1-t) * ut
                #                  (1-t)
                #               |----↓----|
                # 0 ---------- t --------- 1
                # x0---------- xt--------- x1
                #               |--↑--|
                #                 Δt
                stride_gt = torch.where((1-t-delta_t_for_gt) < 0, 1-t, delta_t_for_gt).detach()
                stride_pred = torch.where((1-t-delta_t_for_pred) < 0, 1-t, delta_t_for_pred).detach()

                inputs = xt + stride_gt * ut
                recon = xt.detach() + stride_pred * pred

                g_loss, g_log = disc(
                    inputs.detach()[~drop_ids],
                    recon[~drop_ids],
                    (t + stride_gt)[~drop_ids], 
                    (t + stride_pred)[~drop_ids], 
                    y[~drop_ids], 
                    main_loss,
                    optimizer_idx=0, # 0 - generator
                    global_step=train_steps,
                    last_layer=model.module.module.get_last_layer(),
                    )
                loss = main_loss + g_loss
                opt.zero_grad()
                accelerator.backward(loss)
                if 'max_grad_norm' in train_config['optimizer']:
                    if accelerator.sync_gradients:
                        accelerator.clip_grad_norm_(model.parameters(), train_config['optimizer']['max_grad_norm'])

                # because step_dis in every two step, we need * 2 to log, then averaged by total step
                running_grad_norm += get_grad_norm(model) * 2 if train_steps >= disc.module.discriminator_iter_start else 1
                running_g_loss += g_log['g_loss'] * 2 if train_steps >= disc.module.discriminator_iter_start else 1
                running_d_weight += g_log['d_weight'] * 2 if train_steps >= disc.module.discriminator_iter_start else 1
                running_gen_step_disc_factor += g_log['gen_step_disc_factor'] * 2 if train_steps >= disc.module.discriminator_iter_start else 1

                opt.step()
                update_ema(ema, model.module)

            if step_dis:                
                stride_gt = torch.where((1-t-delta_t_for_gt) < 0, 1-t, delta_t_for_gt)
                stride_pred = torch.where((1-t-delta_t_for_pred) < 0, 1-t, delta_t_for_pred)

                inputs = xt + stride_gt * ut
                recon = xt + stride_pred * pred
                
                g_loss, g_log = disc(
                    inputs.detach()[~drop_ids],
                    recon.detach()[~drop_ids],
                    (t + stride_gt)[~drop_ids], 
                    (t + stride_pred)[~drop_ids], 
                    y[~drop_ids], 
                    main_loss,
                    optimizer_idx=1, # 1 - discriminator
                    global_step=train_steps,
                    last_layer=model.module.module.get_last_layer(),
                    )
                disc_opt.zero_grad()
                accelerator.backward(g_loss)

                # because step_dis in each two step, we need * 2 to log, then averaged by total step
                running_disc_grad_norm += get_grad_norm(disc) * 2 if train_steps >= disc.module.discriminator_iter_start else 1
                running_disc_loss += g_log['disc_loss'] * 2
                running_logits_real += g_log['logits_real'] * 2
                running_logits_fake += g_log['logits_fake'] * 2
                running_disc_step_disc_factor += g_log['disc_step_disc_factor'] * 2

                disc_opt.step()

            # Log loss values:
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
                avg_disc_grad_norm = reduce_over_all_processes(running_disc_grad_norm, log_steps, device=device)
                avg_g_loss = reduce_over_all_processes(running_g_loss, log_steps, device=device)
                avg_d_weight = reduce_over_all_processes(running_d_weight, log_steps, device=device)
                avg_gen_step_disc_factor = reduce_over_all_processes(running_gen_step_disc_factor, log_steps, device=device)
                avg_disc_loss = reduce_over_all_processes(running_disc_loss, log_steps, device=device)
                avg_logits_real = reduce_over_all_processes(running_logits_real, log_steps, device=device)
                avg_logits_fake = reduce_over_all_processes(running_logits_fake, log_steps, device=device)
                avg_disc_step_disc_factor = reduce_over_all_processes(running_disc_step_disc_factor, log_steps, device=device)

                if accelerator.is_main_process:
                    logger.info(
                        f"(step={train_steps:07d}) Train Loss: {avg_loss:.4f}, Train Steps/Sec: {steps_per_sec:.2f}, "
                        f"Grad Norm: {avg_grad_norm:.4f}, Disc Grad Norm: {avg_disc_grad_norm:.4f}, "
                        f"Gen Step Disc Loss: {avg_g_loss:.4f}, Disc Weight: {avg_d_weight:.4f}, Gen Step Disc Factor: {avg_gen_step_disc_factor:.4f}, "
                        f"Disc Step Disc Loss: {avg_disc_loss:.4f}, Logits Real: {avg_logits_real:.4f}, Logits Fake: {avg_logits_fake:.4f}, Disc Step Disc Factor: {avg_disc_step_disc_factor:.4f}"
                        )
                    accelerator.log({
                        "loss": avg_loss, "grad_norm": avg_grad_norm, "disc_grad_norm": avg_disc_grad_norm, 
                        "gan/avg_g_loss": avg_g_loss, "gan/avg_d_weight": avg_d_weight, "gan/avg_gen_step_disc_factor": avg_gen_step_disc_factor, 
                        "gan/avg_disc_loss": avg_disc_loss, "gan/avg_logits_real": avg_logits_real, "gan/avg_logits_fake": avg_logits_fake, "gan/avg_disc_step_disc_factor": avg_disc_step_disc_factor, 
                        }, step=train_steps)
                # Reset monitoring variables:
                running_loss = 0
                running_grad_norm = 0
                running_disc_grad_norm = 0
                running_g_loss = 0
                running_d_weight = 0
                running_gen_step_disc_factor = 0
                running_disc_loss = 0
                running_logits_real = 0
                running_logits_fake = 0
                running_disc_step_disc_factor = 0
                log_steps = 0
                start_time = time()

            # Save checkpoint:
            if train_steps % train_config['train']['ckpt_every'] == 0 and train_steps > 0:
                if accelerator.is_main_process:
                    checkpoint = {
                        "model": model.module.state_dict(),
                        "disc": disc.module.state_dict(),
                        "ema": ema.state_dict(),
                        "opt": opt.state_dict(),
                        "disc_opt": disc_opt.state_dict(),
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