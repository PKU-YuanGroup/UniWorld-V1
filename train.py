# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
A minimal training script for DiT using PyTorch DDP.
"""
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms
from time import time
import argparse
import logging
from packaging import version
import gc
import accelerate
from transformers.integrations import HfDeepSpeedConfig
import deepspeed
import os
import copy
import math
from models import FlowWorld_models, FlowWorld_models_class
from diffusers.models import AutoencoderKL
from scheduler import FlowMatchEulerScheduler as FlowMatchEulerDiscreteScheduler
from diffusers.optimization import get_scheduler
import json
from tqdm import tqdm
import transformers
import diffusers
import shutil
from pathlib import Path
from accelerate.utils import DistributedType, ProjectConfiguration, set_seed
from accelerate import Accelerator
from diffusers.utils import check_min_version, is_wandb_available
from accelerate.logging import get_logger
from utils import EMAModel, get_common_weights, center_crop_arr, compute_density_for_timestep_sampling, ProgressInfo

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.24.0")
logger = get_logger(__name__)
GB = 1024 * 1024 * 1024
#################################################################################
#                             Training Helper Functions                         #
#################################################################################

def create_ema_model(
        args, 
        checkpoint_path, 
        model_cls,
        model_config,
        ema_model_state_dict,
        ds_config=None, 
        rank=-1, 
        ):
    # model_config = AutoConfig.from_pretrained(model_name_or_path)
    ds_config["train_micro_batch_size_per_gpu"] = args.train_batch_size
    ds_config["fp16"]["enabled"] = False
    ds_config["bf16"]["enabled"] = False
    ds_config["gradient_accumulation_steps"] = args.gradient_accumulation_steps
    ds_config["train_batch_size"] = args.train_batch_size * args.gradient_accumulation_steps * args.world_size

    # Note: dschf is defined in function scope to avoid global effects
    # https://huggingface.co/docs/transformers/main_classes/deepspeed#nontrainer-deepspeed-integration
    logging.info(f'EMA deepspeed config {ds_config}')
    if ds_config is not None and ds_config["zero_optimization"]["stage"] == 3:
        dschf = HfDeepSpeedConfig(ds_config)
    else:
        dschf = None
            
    if checkpoint_path:
        ema_model_path = os.path.join(checkpoint_path, "model_ema")
        if os.path.exists(ema_model_path):
            ema_model = EMAModel.from_pretrained(ema_model_path, model_cls=model_cls)
            logger.info(f'Successully resume EMAModel from {ema_model_path}', main_process_only=True)
    else:
        # we load weights from original model instead of deepcopy
        model = model_cls.from_config(model_config)
        model.load_state_dict(ema_model_state_dict, strict=True)
        model = model.eval()
        model.requires_grad_(False)
        ema_model = EMAModel(
            model, decay=args.ema_decay, update_after_step=args.ema_start_step,
            model_cls=model_cls, model_config=model_config
            )
        logger.info(f'Successully deepcopy EMAModel from model', main_process_only=True)
    try:
        ema_model.model, _, _, _ = deepspeed.initialize(model=ema_model.model, config_params=ds_config)
    except Exception as e:
        print(e)
        raise ValueError(f'rank {rank}, error: {e}')
    return ema_model

#################################################################################
#                                  Training Loop                                #
#################################################################################

def main(args):
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."

    logging_dir = Path(args.output_dir, args.logging_dir)
    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )
    if args.report_to == "wandb":
        if not is_wandb_available():
            raise ImportError("Make sure to install wandb if you want to use it for logging during training.")
        wandb_init_kwargs = {"wandb": {"name": args.log_name or args.proj_name or args.output_dir}}
    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        accelerator.init_trackers(os.path.basename(args.proj_name or args.output_dir), config=vars(args), 
                                  init_kwargs=wandb_init_kwargs if args.report_to == "wandb" else None)

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed, device_specific=True)


    # Handle the repository creation
    if args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16


    # Enable TF32 for faster training on Ampere GPUs
    # https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    # =======================================================================================================
    # STEP 0: Resume parameter
    checkpoint_path, global_step = None, 0
    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            checkpoint_path = args.resume_from_checkpoint
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            checkpoint_path = os.path.join(args.output_dir, dirs[-1]) if len(dirs) > 0 else None

        if checkpoint_path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
        else:
            accelerator.print(f"Resuming from checkpoint {checkpoint_path}")
            global_step = int(checkpoint_path.split("-")[-1])
    # =======================================================================================================

    # =======================================================================================================
    # STEP 0: Create model
    assert args.image_size % 8 == 0, "Image size must be divisible by 8 (for the VAE encoder)."
    latent_size = args.image_size // 8
    if checkpoint_path:
        model = FlowWorld_models_class[args.model].from_pretrained(os.path.join(checkpoint_path, "model"))
        logger.info(f'Successully resume model from {os.path.join(checkpoint_path, "model")}', main_process_only=True)
    else:
        model = FlowWorld_models[args.model](
            input_size=latent_size,
            num_classes=args.num_classes, 
            learn_sigma=False
        )
    # use pretrained model?
    if checkpoint_path is None and args.pretrained:
        model_state_dict = model.state_dict()
        logger.info(f'Load from {args.pretrained}')
        if args.pretrained.endswith('.safetensors'):  
            # --pretrained path/to/.safetensors
            from safetensors.torch import load_file as safe_load
            pretrained_checkpoint = safe_load(args.pretrained, device="cpu")
            checkpoint = get_common_weights(pretrained_checkpoint, model_state_dict)
            missing_keys, unexpected_keys = model.load_state_dict(checkpoint, strict=True)
        elif os.path.isdir(args.pretrained):
            # --pretrained path/to/model  or  path/to/model_ema  # must have config.json and .safetensors
            pretrained_model = FlowWorld_models_class[args.model].from_pretrained(args.pretrained)
            pretrained_checkpoint = pretrained_model.state_dict()
            checkpoint = get_common_weights(pretrained_checkpoint, model_state_dict)
            missing_keys, unexpected_keys = model.load_state_dict(checkpoint, strict=True)
            del pretrained_checkpoint, pretrained_model
            gc.collect()
        else:
            # --pretrained path/to/.pth or .pt or some other format
            pretrained_checkpoint = torch.load(args.pretrained, map_location='cpu')
            if 'model' in checkpoint:
                pretrained_checkpoint = pretrained_checkpoint['model']
            checkpoint = get_common_weights(pretrained_checkpoint, model_state_dict)
            missing_keys, unexpected_keys = model.load_state_dict(checkpoint, strict=True)
        logger.info(f'missing_keys {len(missing_keys)} {missing_keys}, unexpected_keys {len(unexpected_keys)}')
        logger.info(f'Successfully load {len(model_state_dict) - len(missing_keys)}/{len(model_state_dict)} keys from {args.pretrained}!')
        del model_state_dict, checkpoint, missing_keys, unexpected_keys
        gc.collect()
    
        # `accelerate` 0.16.0 will have better support for customized saving
    if version.parse(accelerate.__version__) >= version.parse("0.16.0"):
        # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
        def save_model_hook(models, weights, output_dir):
            if accelerator.is_main_process:
                for i, model in enumerate(models):
                    model.save_pretrained(os.path.join(output_dir, "model"))
                    if weights:  # Don't pop if empty
                        # make sure to pop weight so that corresponding model is not saved again
                        weights.pop()
            if args.use_ema:
                ema_model.save_pretrained(os.path.join(output_dir, "model_ema"))

        accelerator.register_save_state_pre_hook(save_model_hook)
        
    if args.gradient_checkpointing:
        model.gradient_checkpointing = True
    logger.info(f"Load diffusion model finish, memory_allocated: {torch.cuda.memory_allocated()/GB:.2f} GB", main_process_only=True)
    # =======================================================================================================



    # =======================================================================================================
    # STEP 2: Create EMAModel
    args.world_size = accelerator.num_processes
    if args.use_ema:
        ema_model_state_dict = model.state_dict()
        with open(args.ema_deepspeed_config_file, 'r') as f:
            ds_config = json.load(f)
        ema_model = create_ema_model(
            args, checkpoint_path, model_cls=FlowWorld_models_class[args.model], model_config=model.config, 
            ema_model_state_dict=ema_model_state_dict, ds_config=ds_config, rank=accelerator.process_index
            )

        logger.info(f"Load ema model finish, memory_allocated: {torch.cuda.memory_allocated()/GB:.2f} GB", main_process_only=True)
    # =======================================================================================================


    # =======================================================================================================
    # STEP 3: Create FlowMatching Scheduler
    noise_scheduler = FlowMatchEulerDiscreteScheduler(weighting_scheme=args.weighting_scheme, sigma_eps=0.0)
    # =======================================================================================================

    # =======================================================================================================
    # STEP 4: Create VAE
    # vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}", cache_dir='./cache_dir').to(accelerator.device)
    vae = AutoencoderKL.from_pretrained(f"cache_dir/models--stabilityai--sd-vae-ft-ema/snapshots/f04b2c4b98319346dad8c65879f680b1997b204a")
    vae.requires_grad_(False)
    vae.to(accelerator.device, dtype=torch.float32)
    # =======================================================================================================

    # =======================================================================================================
    # STEP 5: Setup data
    transform = transforms.Compose([
        transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, args.image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
    ])
    train_dataset = ImageFolder(args.data_path, transform=transform)

    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps
    args.total_batch_size = total_batch_size
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )
    # =======================================================================================================

    # =======================================================================================================
    # STEP 6: LR_Scheduler
    # Scheduler and math around the number of training steps.
    override_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        override_max_train_steps = True
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0)
    lr_scheduler = get_scheduler(
            args.lr_scheduler,
            optimizer=optimizer,
            num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
            num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
        )
    # =======================================================================================================

    # =======================================================================================================
    # STEP 7: Prepare everything with our `accelerator`.
    logger.info(f"Before accelerator.prepare, memory_allocated: {torch.cuda.memory_allocated()/GB:.2f} GB", main_process_only=True)
    model_config = model.config
    try:
        model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            model, optimizer, train_dataloader, lr_scheduler
        )
    except Exception as e:
        print(e)
        raise ValueError(f'rank {accelerator.process_index}, error: {e}')
    if checkpoint_path:
        accelerator.load_state(checkpoint_path)
    logger.info(f"After accelerator.prepare, memory_allocated: {torch.cuda.memory_allocated()/GB:.2f} GB", main_process_only=True)

    model.train()
    # =======================================================================================================
    # STEP 8: Train!
    logger.info("***** Running training *****")
    logger.info(f"  Model = {model}")
    logger.info(f'  Model config = {model_config}')
    logger.info(f"  Args = {args}")
    logger.info(f"  Noise_scheduler = {noise_scheduler}")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    logger.info(f"  Total optimization steps (num_update_steps_per_epoch) = {num_update_steps_per_epoch}")
    logger.info(f"  Total training parameters = {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e9} B")
    logger.info(f"  AutoEncoder; Dtype = {vae.dtype}; Parameters = {sum(p.numel() for p in vae.parameters()) / 1e9} B")
    if args.use_ema:
        logger.info(f"  EMA model = {type(ema_model.model)}; Dtype = {ema_model.model.dtype}; Parameters = {sum(p.numel() for p in ema_model.model.parameters()) / 1e9} B")

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if override_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )
    progress_info = ProgressInfo(global_step, train_loss=0.0)

    def train_one_epoch():
        for x, y in train_dataloader:
            x = x.to(accelerator.device, non_blocking=True, dtype=vae.dtype)
            y = y.to(accelerator.device, non_blocking=True)
            with torch.no_grad():
                # Map input images to latent space + normalize latents:
                model_input = vae.encode(x).latent_dist.sample().mul_(0.18215)
            noise = torch.randn_like(model_input)
            bsz = model_input.shape[0]
            # Sample a random timestep for each image
            # for weighting schemes where we sample timesteps non-uniformly
            sigmas = noise_scheduler.compute_density_for_sigma_sampling(
                batch_size=bsz,
                logit_mean=args.logit_mean,
                logit_std=args.logit_std,
                mode_scale=args.mode_scale,
            ).to(device=accelerator.device)
            timesteps = sigmas.clone() * 1000.0

            while sigmas.ndim < model_input.ndim:
                sigmas = sigmas.unsqueeze(-1)

            noisy_model_input = noise_scheduler.add_noise(model_input, sigmas, noise)

            model_kwargs = dict(y=y)
            noisy_model_input = noisy_model_input.to(weight_dtype)
            # print(f'noisy_model_input: {noisy_model_input.dtype}, timesteps: {timesteps.dtype}')
            with accelerator.accumulate(model):
                model_pred = model(noisy_model_input, timesteps, **model_kwargs)

            # these weighting schemes use a uniform timestep sampling
            # and instead post-weight the loss
            weighting = noise_scheduler.compute_loss_weighting_for_sd3(sigmas=sigmas)
            # print(sigmas, weighting)
            # flow matching loss
            target = noise - model_input
            # Compute regular loss.
            loss_mse = (weighting.float() * (model_pred.float() - target.float()) ** 2).reshape(target.shape[0], -1)
            loss = loss_mse.mean()
            accelerator.backward(loss)
            optimizer.step()
            avg_loss_list = accelerator.gather(loss)
            progress_info.train_loss += avg_loss_list.mean().detach().item() / args.gradient_accumulation_steps
            optimizer.zero_grad()
            lr_scheduler.step()

            if accelerator.sync_gradients:
                if args.use_ema and progress_info.global_step % args.ema_update_freq == 0:
                    ema_model.step(model.parameters())
                    cur_decay_value = ema_model.cur_decay_value
                progress_bar.update(1)
                progress_info.global_step += 1
                
                train_loss = progress_info.train_loss
                log_dict = {"train_loss": train_loss, "lr": lr_scheduler.get_last_lr()[0]}
                if args.use_ema and (progress_info.global_step - 1) % args.ema_update_freq == 0:
                    log_dict.update(dict(cur_decay_value=cur_decay_value))
                accelerator.log(log_dict, step=progress_info.global_step)
                progress_info.train_loss = 0.0

                # DeepSpeed requires saving weights on every device; saving weights only on the main process would cause issues.
                if accelerator.distributed_type == DistributedType.DEEPSPEED or accelerator.is_main_process:
                    if progress_info.global_step % args.checkpointing_steps == 0:
                        # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                        if accelerator.is_main_process and args.checkpoints_total_limit is not None:
                            checkpoints = os.listdir(args.output_dir)
                            checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                            # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                            if len(checkpoints) >= args.checkpoints_total_limit:
                                num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                                removing_checkpoints = checkpoints[0:num_to_remove]
                                logger.info(f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints")
                                logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                                for removing_checkpoint in removing_checkpoints:
                                    removing_checkpoint = os.path.join(args.output_dir, removing_checkpoint)
                                    shutil.rmtree(removing_checkpoint)

                        save_path = os.path.join(args.output_dir, f"checkpoint-{progress_info.global_step}")
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")

                logs = {"step_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
                progress_bar.set_postfix(**logs)
            if progress_info.global_step >= args.max_train_steps:
                return True
        return False

    for _ in range(args.num_train_epochs):
        if train_one_epoch():
            break


    # DeepSpeed requires saving weights on every device; saving weights only on the main process would cause issues.
    if accelerator.distributed_type == DistributedType.DEEPSPEED or accelerator.is_main_process:
        save_path = os.path.join(args.output_dir, f"checkpoint-{progress_info.global_step}")
        accelerator.save_state(save_path)
        logger.info(f"Saved state to {save_path}")

    accelerator.wait_for_everyone()
    accelerator.end_training()



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    # deepspeed
    parser.add_argument(
        "--ema_deepspeed_config_file", type=str, default='scripts/accelerate_configs/zero3.json', 
        help="deepspeed config file for EMA model"
        )
    
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--model", type=str, choices=list(FlowWorld_models.keys()), default="DiT-XL/2")
    parser.add_argument("--image_size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--num_classes", type=int, default=1000)
    parser.add_argument("--num_train_epochs", type=int, default=1400)
    parser.add_argument("--train_batch_size", type=int, default=256)
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="ema")  # Choice doesn't affect training
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--gradient_checkpointing", action="store_true", help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.")
    parser.add_argument("--use_ema", action="store_true", help="Whether to use EMA model.")
    parser.add_argument("--ema_decay", type=float, default=0.99)
    parser.add_argument("--ema_update_freq", type=int, default=100)
    parser.add_argument("--ema_start_step", type=int, default=0)
    parser.add_argument("--pretrained", type=str, default=None)

    parser.add_argument("--max_train_steps", type=int, default=None, help="Total number of training steps to perform.  If provided, overrides num_train_epochs.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Initial learning rate (after the potential warmup period) to use.")
    parser.add_argument("--lr_warmup_steps", type=int, default=0, help="Number of steps for the warmup in the lr scheduler.")
    parser.add_argument("--lr_scheduler", type=str, default="constant_with_warmup",
                        help=(
                            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
                            ' "constant", "constant_with_warmup"]'
                        ),
                        )

    parser.add_argument("--weighting_scheme", type=str, default="logit_normal", choices=["sigma_sqrt", "logit_normal", "mode", "cosmap"])
    parser.add_argument("--logit_mean", type=float, default=0.0, help="mean to use when using the `'logit_normal'` weighting scheme.")
    parser.add_argument("--logit_std", type=float, default=1.0, help="std to use when using the `'logit_normal'` weighting scheme.")
    parser.add_argument("--mode_scale", type=float, default=1.29, help="Scale of mode weighting scheme. Only effective when using the `'mode'` as the `weighting_scheme`.")
    
    parser.add_argument("--seed", type=int, default=42, help="A seed for reproducible training.")
    parser.add_argument("--output_dir", type=str, default=None, help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--proj_name", type=str, default=None, help="Custom project names for the runs in W&B logger, default to output_dir.")
    parser.add_argument("--log_name", type=str, default=None, help="Custom run names for the runs in W&B logger, default to proj_name or output_dir.")
    parser.add_argument("--checkpoints_total_limit", type=int, default=None, help=("Max number of checkpoints to store."))
    parser.add_argument("--checkpointing_steps", type=int, default=1000,
                        help=(
                            "Save a checkpoint of the training state every X updates. These checkpoints can be used both as final"
                            " checkpoints in case they are better than the last checkpoint, and are also suitable for resuming"
                            " training using `--resume_from_checkpoint`."
                        ),
                        )
    parser.add_argument("--resume_from_checkpoint", type=str, default=None,
                        help=(
                            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
                            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
                        ),
                        )
    parser.add_argument("--logging_dir", type=str, default="logs",
                    help=(
                        "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
                        " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
                    ),
                    )
    parser.add_argument("--report_to", type=str, default="wandb",
                        help=(
                            'The integration to report the results and logs to. Supported platforms are `"wandb"` (default)'
                            ', `"tensorboard"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
                        ),
                        )    
    parser.add_argument("--allow_tf32", action="store_true",
                        help=(
                            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
                            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
                        ),
                        )
    parser.add_argument("--mixed_precision", type=str, default='bf16', choices=["no", "fp16", "bf16"],
                        help=(
                            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
                            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
                            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
                        ),
                        )
    args = parser.parse_args()
    main(args)
