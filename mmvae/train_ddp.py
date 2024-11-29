import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from torch.utils.data import DataLoader, DistributedSampler, Subset
import argparse
import logging
from colorlog import ColoredFormatter
import tqdm
from itertools import chain
import wandb
import random
import math
import numpy as np
from torchvision.datasets import ImageFolder
from torchvision import transforms
from pathlib import Path
from PIL import Image
from einops import rearrange
import torchvision
from torch import nn
from causalvideovae.model import *
from causalvideovae.model.ema_model import EMA
from causalvideovae.dataset.ddp_sampler import CustomDistributedSampler
from causalvideovae.dataset.video_dataset import TrainVideoDataset, ValidVideoDataset
from causalvideovae.model.utils.module_utils import resolve_str_to_obj
from causalvideovae.utils.video_utils import tensor_to_image
from causalvideovae.model.losses import ClipLoss
try:
    import lpips
except:
    raise Exception("Need lpips to valid.")

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def ddp_setup():
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

def setup_logger(rank):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    formatter = logging.Formatter(
        f"[rank{rank}] %(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.DEBUG)
    stream_handler.setFormatter(formatter)
    
    logger.addHandler(stream_handler)
    return logger

def check_unused_params(model):
    unused_params = []
    for name, param in model.named_parameters():
        if param.grad is None:
            unused_params.append(name)
    return unused_params

def set_requires_grad_optimizer(optimizer, requires_grad):
    for param_group in optimizer.param_groups:
        for param in param_group["params"]:
            param.requires_grad = requires_grad

def total_params(model):
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params_in_millions = total_params / 1e6
    return int(total_params_in_millions)

def get_exp_name(args):
    return f"{args.exp_name}-lr{args.lr:.2e}-bs{args.batch_size}-rs{args.resolution}-sr{args.sample_rate}-fr{args.num_frames}"

def set_train(modules):
    for module in modules:
        module.train()

def set_eval(modules):
    for module in modules:
        module.eval()

def set_modules_requires_grad(modules, requires_grad):
    for module in modules:
        module.requires_grad_(requires_grad)


def unwrap_model(model):
    if hasattr(model, 'module'):
        return model.module
    else:
        return model


def accuracy(output, target, topk=(1,)):
    pred = output.topk(max(topk), 1, True, True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    return [float(correct[:k].reshape(-1).float().sum(0, keepdim=True).cpu().numpy()) for k in topk]



def save_checkpoint(
    epoch,
    current_step,
    optimizer_state,
    state_dict,
    scaler_state,
    sampler_state,
    checkpoint_dir,
    filename="checkpoint.ckpt",
    ema_state_dict={},
):
    filepath = checkpoint_dir / Path(filename)
    torch.save(
        {
            "epoch": epoch,
            "current_step": current_step,
            "optimizer_state": optimizer_state,
            "state_dict": state_dict,
            "ema_state_dict": ema_state_dict,
            "scaler_state": scaler_state,
            "sampler_state": sampler_state,
        },
        filepath,
    )
    return filepath


def valid(global_rank, rank, model, val_dataloader, precision, args):
    if args.train_image and args.eval_lpips:
        lpips_model = lpips.LPIPS(net="alex", spatial=True)
        lpips_model.to(rank)
        lpips_model = DDP(lpips_model, device_ids=[rank])
        lpips_model.requires_grad_(False)
        lpips_model.eval()

    bar = None
    if global_rank == 0:
        bar = tqdm.tqdm(total=len(val_dataloader), desc="Validation...")

    psnr_list = []
    lpips_list = []
    video_log = []
    num_video_log = args.eval_num_video_log

    sample_num = 0
    ce_loss_list = [] 
    acc_list = []  

    num_classes = 1000
    classnames = torch.arange(num_classes).long().to(rank)
    classnames = classnames.reshape(-1, 1)
    clip_acc1_list, clip_acc5_list = [], []
    enc_acc1_list, enc_acc5_list = [], []
    if args.train_image and args.train_text:
        with torch.no_grad():
            classifier_enc = model.module.get_encoder_text_features(classnames).T
            if model.module.use_clip_loss:
                classifier_clip = model.module.get_clip_text_features(classnames).T

    ce_loss_fun = nn.CrossEntropyLoss()
    with torch.no_grad():
        for batch_idx, batch in enumerate(val_dataloader):
            inputs, labels = batch
            if labels.ndim == 1:
                labels = labels.reshape(-1, 1)
            inputs = inputs.to(rank, non_blocking=True)
            labels = labels.to(rank, non_blocking=True)

            sample_num += inputs.shape[0]

            with torch.amp.autocast("cuda", dtype=precision):
                output = model(inputs, labels)
                video_recon, text_logit = output.sample
                text_logit = text_logit.squeeze(1) if args.train_text else None

            if args.train_text:
                pred_classes = torch.max(text_logit, dim=1)[1]
                labels = labels.squeeze(1) if text_logit is not None else None
                ce_loss_list.append(ce_loss_fun(text_logit, labels).cpu().item())
                acc_list.append(torch.eq(pred_classes, labels).sum().cpu().item() / inputs.shape[0])

            if args.train_image and args.train_text:
                enc_image_features = output.enc_features[0]
                enc_logits = 100. * enc_image_features @ classifier_enc
                enc_acc1, enc_acc5 = accuracy(enc_logits, labels, topk=(1, 5))
                enc_acc1_list.append(enc_acc1)
                enc_acc5_list.append(enc_acc5)

            if args.clip_loss:
                image_features = output.clip_feature[0]
                logits = 100. * image_features @ classifier_clip
                acc1, acc5 = accuracy(logits, labels, topk=(1, 5))
                clip_acc1_list.append(acc1)
                clip_acc5_list.append(acc5)
            
            if args.train_image:
                # Upload videos
                if global_rank == 0:
                    for i in range(len(video_recon)):
                        if num_video_log <= 0:
                            break
                        video = tensor_to_image(video_recon[i])
                        video_log.append(video)
                        num_video_log -= 1

                # Calculate PSNR
                mse = torch.mean(torch.square(inputs - video_recon), dim=(1, 2, 3))
                psnr = 20 * torch.log10(1 / torch.sqrt(mse))
                psnr = psnr.mean().detach().cpu().item()

                # Calculate LPIPS
                if args.eval_lpips:
                    lpips_score = (
                        lpips_model.forward(inputs, video_recon)
                        .mean()
                        .detach()
                        .cpu()
                        .item()
                    )
                    lpips_list.append(lpips_score)

                psnr_list.append(psnr)
            
            if global_rank == 0:
                bar.update()
            # Release gpus memory
            torch.cuda.empty_cache()
    return psnr_list, lpips_list, video_log, ce_loss_list, acc_list, clip_acc1_list, clip_acc5_list, enc_acc1_list, enc_acc5_list

def gather_valid_result(
    psnr_list, lpips_list, video_log_list, ce_loss_list, acc_list, 
    clip_acc1_list, clip_acc5_list, enc_acc1_list, enc_acc5_list, rank, world_size
):
    gathered_psnr_list = [None for _ in range(world_size)]
    gathered_lpips_list = [None for _ in range(world_size)]
    gathered_video_logs = [None for _ in range(world_size)]
    gathered_ce_loss_list = [None for _ in range(world_size)]
    gathered_acc_list = [None for _ in range(world_size)]
    gathered_clip_acc1_list = [None for _ in range(world_size)]
    gathered_clip_acc5_list = [None for _ in range(world_size)]
    gathered_enc_acc1_list = [None for _ in range(world_size)]
    gathered_enc_acc5_list = [None for _ in range(world_size)]
    
    dist.all_gather_object(gathered_psnr_list, psnr_list)
    dist.all_gather_object(gathered_lpips_list, lpips_list)
    dist.all_gather_object(gathered_video_logs, video_log_list)
    dist.all_gather_object(gathered_ce_loss_list, ce_loss_list)
    dist.all_gather_object(gathered_acc_list, acc_list)
    dist.all_gather_object(gathered_clip_acc1_list, clip_acc1_list)
    dist.all_gather_object(gathered_clip_acc5_list, clip_acc5_list)
    dist.all_gather_object(gathered_enc_acc1_list, enc_acc1_list)
    dist.all_gather_object(gathered_enc_acc5_list, enc_acc5_list)
    return (
        np.array(gathered_psnr_list).mean(),
        np.array(gathered_lpips_list).mean(),
        list(chain(*gathered_video_logs)),
        np.array(gathered_ce_loss_list).mean(),
        np.array(gathered_acc_list).mean(),
        np.array(gathered_clip_acc1_list).mean(),
        np.array(gathered_clip_acc5_list).mean(),
        np.array(gathered_enc_acc1_list).mean(),
        np.array(gathered_enc_acc5_list).mean(),
    )

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


def train(args):
    # setup logger
    ddp_setup()
    rank = int(os.environ["LOCAL_RANK"])
    global_rank = dist.get_rank()
    logger = setup_logger(rank)

    # init
    ckpt_dir = Path(args.ckpt_dir) / Path(get_exp_name(args))
    if global_rank == 0:
        try:
            ckpt_dir.mkdir(exist_ok=False, parents=True)
        except:
            logger.warning(f"`{ckpt_dir}` exists!")
    dist.barrier()

    # load generator model
    model_cls = ModelRegistry.get_model(args.model_name)

    if not model_cls:
        raise ModuleNotFoundError(
            f"`{args.model_name}` not in {str(ModelRegistry._models.keys())}."
        )

    if args.pretrained_model_name_or_path is not None:
        if global_rank == 0:
            logger.warning(
                f"You are loading a checkpoint from `{args.pretrained_model_name_or_path}`."
            )
        model = model_cls.from_pretrained(
            args.pretrained_model_name_or_path,
            ignore_mismatched_sizes=args.ignore_mismatched_sizes,
            low_cpu_mem_usage=False,
            device_map=None,
        )
    else:
        if global_rank == 0:
            logger.warning(f"Model will be inited randomly.")
        print('args.model_config', args.model_config)
        model = model_cls.from_config(args.model_config)
    
    if global_rank == 0:
        logger.warning("Connecting to WANDB...")
        model_config = dict(**model.config)
        args_config = dict(**vars(args))
        if 'resolution' in model_config:
            del model_config['resolution']
        if 'train_image' in model_config:
            del model_config['train_image']
        if 'train_text' in model_config:
            del model_config['train_text']
        
        wandb.init(
            project=os.environ.get("WANDB_PROJECT", "causalvideovae"),
            config=dict(**model_config, **args_config),
            name=get_exp_name(args),
        )
        
    dist.barrier()
    
    if args.train_image:
        # load discriminator model
        disc_cls = resolve_str_to_obj(args.disc_cls, append=False)
        logger.warning(
            f"disc_class: {args.disc_cls} perceptual_weight: {args.perceptual_weight}  loss_type: {args.loss_type}"
        )
        disc = disc_cls(
            disc_start=args.disc_start,
            disc_weight=args.disc_weight,
            kl_weight=args.kl_weight,
            logvar_init=args.logvar_init,
            perceptual_weight=args.perceptual_weight,
            loss_type=args.loss_type,
            wavelet_weight=args.wavelet_weight
        )

    # DDP
    model = model.to(rank)
    
    if args.enable_tiling:
        model.enable_tiling()
    
    model = DDP(
        model, device_ids=[rank], find_unused_parameters=args.find_unused_parameters
    )
    if args.train_image:
        disc = disc.to(rank)
        disc = DDP(
            disc, device_ids=[rank], find_unused_parameters=args.find_unused_parameters
        )

    # load dataset
    transform = transforms.Compose([
        transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, args.resolution)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
    ])
    dataset = ImageFolder(args.video_path, transform=transform)

    ddp_sampler = CustomDistributedSampler(dataset)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=ddp_sampler,
        pin_memory=True,
        num_workers=args.dataset_num_worker,
    )


    
    val_dataset = ImageFolder(args.eval_video_path, transform=transform)
    indices = range(args.eval_subset_size)
    val_dataset = Subset(val_dataset, indices=indices)
    val_sampler = CustomDistributedSampler(val_dataset)
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=args.eval_batch_size,
        sampler=val_sampler,
        pin_memory=True,
    )

    # optimizer
    modules_to_train = [module for module in model.module.get_decoder()]
    if args.freeze_encoder:
        for module in model.module.get_encoder():
            module.eval()
            module.requires_grad_(False)
        logger.info("Encoder is freezed!")
    else:
        modules_to_train += [module for module in model.module.get_encoder()]

    if args.clip_loss:
        assert args.train_image and args.train_text
        if args.freeze_clip_param:
            for module in model.module.get_clip_param():
                module.eval()
                module.requires_grad_(False)
            logger.info("Clip_param is freezed!")
        else:
            modules_to_train += [module for module in model.module.get_clip_param()]

    parameters_to_train = []
    for module in modules_to_train:
        parameters_to_train += list(filter(lambda p: p.requires_grad, module.parameters()))
    gen_optimizer = torch.optim.AdamW(parameters_to_train, lr=args.lr, weight_decay=args.weight_decay)

    if args.train_image:
        disc_optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, disc.module.discriminator.parameters()), lr=args.lr, weight_decay=args.weight_decay
        )

    # AMP scaler
    scaler = torch.amp.GradScaler('cuda')
    precision = torch.bfloat16
    if args.mix_precision == "fp16":
        precision = torch.float16
    elif args.mix_precision == "fp32":
        precision = torch.float32
    
    # load from checkpoint
    start_epoch = 0
    current_step = 0
    if args.resume_from_checkpoint:
        if not os.path.isfile(args.resume_from_checkpoint):
            raise Exception(
                f"Make sure `{args.resume_from_checkpoint}` is a ckpt file."
            )
        checkpoint = torch.load(args.resume_from_checkpoint, map_location="cpu")
        model.module.load_state_dict(checkpoint["state_dict"]["gen_model"], strict=False)
        
        # resume optimizer
        if not args.not_resume_optimizer:
            gen_optimizer.load_state_dict(checkpoint["optimizer_state"]["gen_optimizer"])
        
        # resume discriminator
        if not args.not_resume_discriminator:
            disc.module.load_state_dict(checkpoint["state_dict"]["dics_model"])
            disc_optimizer.load_state_dict(checkpoint["optimizer_state"]["disc_optimizer"])
            scaler.load_state_dict(checkpoint["scaler_state"])
        
        # resume data sampler
        ddp_sampler.load_state_dict(checkpoint["sampler_state"])
        
        start_epoch = checkpoint["sampler_state"]["epoch"]
        current_step = checkpoint["current_step"]
        logger.info(
            f"Checkpoint loaded from {args.resume_from_checkpoint}, starting from epoch {start_epoch} step {current_step}"
        )

    if args.ema:
        logger.warning(f"Start with EMA. EMA decay = {args.ema_decay}.")
        ema = EMA(model, args.ema_decay)
        ema.register()

    logger.info("Prepared!")
    dist.barrier()
    if global_rank == 0:
        if args.train_image:
            logger.info(f"Generator:\t\t{total_params(model.module)}M")
            logger.info(f"\t- Encoder:\t{total_params(model.module.encoder):d}M")
            logger.info(f"\t- Decoder:\t{total_params(model.module.decoder):d}M")
            logger.info(f"Discriminator:\t{total_params(disc.module):d}M")
        if args.train_text:
            logger.info(f"\t- TextEncoder:\t{total_params(model.module.text_encoder):d}M")
            logger.info(f"\t- TextDecoder:\t{total_params(model.module.text_decoder):d}M")
        logger.info(f"\t- MMEncoder:\t{total_params(model.module.mm_encoder):d}M")
        logger.info(f"Precision is set to: {args.mix_precision}!")
        logger.info("Start training!")

    # training bar
    bar_desc = "Epoch: {current_epoch}, Loss: {loss}"
    bar = None
    if global_rank == 0:
        max_steps = (
            args.epochs * len(dataloader) if args.max_steps is None else args.max_steps
        )
        bar = tqdm.tqdm(total=max_steps, desc=bar_desc.format(current_epoch=0, loss=0))
        bar.update(current_step)
        logger.warning("Training Details: ")
        logger.warning(f" Max steps: {max_steps}")
        logger.warning(f" Dataset Samples: {len(dataloader)}")
        logger.warning(
            f" Total Batch Size: {args.batch_size} * {os.environ['WORLD_SIZE']}"
        )
    dist.barrier()

    num_epochs = args.epochs
    ce_loss_fun = nn.CrossEntropyLoss()
    clip_loss_fun = ClipLoss(
        local_loss=False,
        gather_with_grad=False,
        cache_labels=True,
        rank=rank,
        world_size=dist.get_world_size(),
        use_horovod=False,
        )
    def update_bar(bar):
        if global_rank == 0:
            bar.desc = bar_desc.format(current_epoch=epoch, loss=f"-")
            bar.update()
    
    # training Loop
    for epoch in range(num_epochs):
        set_train(modules_to_train)
        ddp_sampler.set_epoch(epoch)  # Shuffle data at every epoch
        
        for batch_idx, batch in enumerate(dataloader):
            inputs, labels = batch
            if labels.ndim == 1:
                labels = labels.reshape(-1, 1)
            inputs = inputs.to(rank, non_blocking=True)
            labels = labels.to(rank, non_blocking=True)
            
            if args.train_image:
                # select generator or discriminator
                if (
                    current_step % 2 == 1
                    and current_step >= disc.module.discriminator_iter_start
                ):
                    set_modules_requires_grad(modules_to_train, False)
                    step_gen = False
                    step_dis = True
                else:
                    set_modules_requires_grad(modules_to_train, True)
                    step_gen = True
                    step_dis = False
                
                assert (
                    step_gen or step_dis
                ), "You should backward either Gen. or Dis. in a step."
            else:
                step_gen = True
                step_dis = False

            # forward
            with torch.amp.autocast('cuda', dtype=precision):
                outputs = model(inputs if args.train_image else None, labels if args.train_text else None)
                if args.clip_loss:
                    assert args.train_image and args.train_text
                    image_features, text_features, logit_scale = outputs.clip_feature

                recon, text_logit = outputs.sample
                posterior = outputs.latent_dist
                wavelet_coeffs = None
                if outputs.extra_output is not None and args.wavelet_loss:
                    wavelet_coeffs = outputs.extra_output
            
            # text ce loss
            if args.train_text:
                text_logit = text_logit.squeeze(1)
                labels = labels.squeeze(1)
                ce_loss = ce_loss_fun(text_logit.float(), labels)

            if args.train_image and args.train_text:
                enc_features = outputs.enc_features
                enc_clip_loss = clip_loss_fun(enc_features[0], enc_features[1], logit_scale if args.clip_loss else 1.0, output_dict=False)

            # clip loss
            if args.clip_loss:
                assert args.train_image and args.train_text
                clip_loss = clip_loss_fun(image_features, text_features, logit_scale, output_dict=False)
            
            
            # generator loss
            if step_gen:
                if args.train_image:
                    with torch.amp.autocast('cuda', dtype=precision):
                        g_loss, g_log = disc(
                            inputs,
                            recon,
                            posterior,
                            optimizer_idx=0, # 0 - generator
                            global_step=current_step,
                            last_layer=model.module.get_last_layer(),
                            wavelet_coeffs=wavelet_coeffs,
                            split="train",
                        )
                gen_optimizer.zero_grad()
                g_loss_ = (g_loss if args.train_image else 0.0) + (ce_loss if args.train_text else 0.0) + (clip_loss if args.clip_loss else 0.0)
                scaler.scale(g_loss_).backward()
                scaler.step(gen_optimizer)
                scaler.update()

                # update ema
                if args.ema:
                    ema.update()
                
                # Note: we clamp to 4.6052 = ln(100), as in the original paper.
                if args.clip_loss:
                    with torch.no_grad():
                        unwrap_model(model).clip_head.logit_scale.clamp_(0, math.log(100))
                # log to wandb
                if global_rank == 0 and current_step % args.log_steps == 0:
                    if args.train_image:
                        wandb.log(
                            {"train/generator_loss": g_loss.item()}, step=current_step
                        )
                        wandb.log(
                            {"train/rec_loss": g_log['train/rec_loss']}, step=current_step
                        )
                    sample, sample_t = posterior.sample()
                    if sample is not None:
                        wandb.log(
                            {"train/latents_std": sample.std().item()}, step=current_step
                        )
                    if sample_t is not None:
                        wandb.log(
                            {"train/latents_t_std": sample_t.std().item()}, step=current_step
                        )
                    if args.train_text:
                        wandb.log(
                            {"train/ce_loss": ce_loss.item()}, step=current_step
                        )
                    if args.train_text and args.train_image:
                        wandb.log(
                            {"train/enc_clip_loss": enc_clip_loss.item()}, step=current_step
                        )
                    if args.clip_loss:
                        wandb.log(
                            {"train/clip_loss": clip_loss.item()}, step=current_step
                        )
                        wandb.log(
                            {"train/logit_scale": model.module.clip_head.logit_scale.detach().data.item()}, step=current_step
                        )

            # discriminator loss
            if args.train_image:
                if step_dis:
                    with torch.amp.autocast('cuda', dtype=precision):
                        d_loss, d_log = disc(
                            inputs,
                            recon,
                            posterior,
                            optimizer_idx=1,
                            global_step=current_step,
                            last_layer=None,
                            split="train",
                        )
                    disc_optimizer.zero_grad()
                    scaler.scale(d_loss).backward()
                    scaler.unscale_(disc_optimizer)
                    scaler.step(disc_optimizer)
                    scaler.update()
                    if global_rank == 0 and current_step % args.log_steps == 0:
                        wandb.log(
                            {"train/discriminator_loss": d_loss.item()}, step=current_step
                        )

            update_bar(bar)
            current_step += 1

            # valid model
            
            def valid_model(model, name=""):
                set_eval(modules_to_train)
                psnr_list, lpips_list, video_log, ce_loss_list, acc_list, clip_acc1_list, clip_acc5_list, enc_acc1_list, enc_acc5_list = valid(
                    global_rank, rank, model, val_dataloader, precision, args
                )
                valid_psnr, valid_lpips, valid_video_log, valid_ce_loss, valid_acc, valid_clip_acc1, valid_clip_acc5, valid_enc_acc1, valid_enc_acc5 = gather_valid_result(
                    psnr_list, lpips_list, video_log, ce_loss_list, acc_list, clip_acc1_list, clip_acc5_list, enc_acc1_list, enc_acc5_list, rank, dist.get_world_size()
                )
                if global_rank == 0:
                    name = "_" + name if name != "" else name
                    if args.train_image:
                        valid_video_log_ = torchvision.utils.make_grid(torch.from_numpy(np.stack(valid_video_log)))
                        valid_video_log_ = rearrange(valid_video_log_, 'c h w -> h w c').numpy().astype(np.uint8)
                        wandb.log(
                            {
                                f"val{name}/recon": wandb.Image(
                                    valid_video_log_
                                )
                            },
                            step=current_step,
                        )
                        wandb.log({f"val{name}/psnr": valid_psnr}, step=current_step)
                        wandb.log({f"val{name}/lpips": valid_lpips}, step=current_step)
                    if args.train_text:
                        wandb.log({f"val{name}/ce_loss": valid_ce_loss}, step=current_step)
                        wandb.log({f"val{name}/acc": valid_acc}, step=current_step)
                    
                    if args.train_image and args.train_text:
                        wandb.log({f"val{name}/enc_acc1": valid_enc_acc1}, step=current_step)
                        wandb.log({f"val{name}/enc_acc5": valid_enc_acc5}, step=current_step)

                    if args.clip_loss:
                        wandb.log({f"val{name}/clip_acc1": valid_clip_acc1}, step=current_step)
                        wandb.log({f"val{name}/clip_acc5": valid_clip_acc5}, step=current_step)

                    logger.info(f"{name} Validation done.")

            # if current_step % args.eval_steps == 0:
            if current_step % args.eval_steps == 0 or current_step == 1:
                if global_rank == 0:
                    logger.info("Starting validation...")
                valid_model(model)
                if args.ema:
                    ema.apply_shadow()
                    valid_model(model, "ema")
                    ema.restore()

            # save checkpoint
            if current_step % args.save_ckpt_step == 0 and global_rank == 0:
                file_path = save_checkpoint(
                    epoch,
                    current_step,
                    {
                        "gen_optimizer": gen_optimizer.state_dict(),
                        "disc_optimizer": disc_optimizer.state_dict() if args.train_image else {},
                    },
                    {
                        "gen_model": model.module.state_dict(),
                        "dics_model": disc.module.state_dict() if args.train_image else {},
                    },
                    scaler.state_dict(),
                    ddp_sampler.state_dict(),
                    ckpt_dir,
                    f"checkpoint-{current_step}.ckpt",
                    ema_state_dict=ema.shadow if args.ema else {},
                )
                logger.info(f"Checkpoint has been saved to `{file_path}`.")
                
    # end training
    dist.destroy_process_group()

def main():
    parser = argparse.ArgumentParser(description="Distributed Training")

    # mmloss setting
    parser.add_argument("--clip_loss", action="store_true", help="")
    parser.add_argument("--freeze_clip_param", action="store_true", help="")
    parser.add_argument("--train_image", action="store_true", help="")
    parser.add_argument("--train_text", action="store_true", help="")

    # Exp setting
    parser.add_argument(
        "--exp_name", type=str, default="test", help="number of epochs to train"
    )
    parser.add_argument("--seed", type=int, default=1234, help="seed")
    # Training setting
    parser.add_argument(
        "--epochs", type=int, default=10, help="number of epochs to train"
    )
    parser.add_argument(
        "--max_steps", type=int, default=None, help="number of epochs to train"
    )
    parser.add_argument("--save_ckpt_step", type=int, default=1000, help="")
    parser.add_argument("--ckpt_dir", type=str, default="./results/", help="")
    parser.add_argument(
        "--batch_size", type=int, default=1, help="batch size for training"
    )
    parser.add_argument("--lr", type=float, default=1e-5, help="learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="weight decay")
    parser.add_argument("--log_steps", type=int, default=5, help="log steps")
    parser.add_argument("--freeze_encoder", action="store_true", help="")
    parser.add_argument("--clip_grad_norm", type=float, default=1e5, help="")

    # Data
    parser.add_argument("--video_path", type=str, default=None, help="")
    parser.add_argument("--num_frames", type=int, default=17, help="")
    parser.add_argument("--resolution", type=int, default=256, help="")
    parser.add_argument("--sample_rate", type=int, default=2, help="")
    parser.add_argument("--dynamic_sample", action="store_true", help="")
    # Generator model
    parser.add_argument("--ignore_mismatched_sizes", action="store_true", help="")
    parser.add_argument("--find_unused_parameters", action="store_true", help="")
    parser.add_argument(
        "--pretrained_model_name_or_path", type=str, default=None, help=""
    )
    parser.add_argument("--model_name", type=str, default=None, help="")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None, help="")
    parser.add_argument("--not_resume_training_process", action="store_true", help="")
    parser.add_argument("--enable_tiling", action="store_true", help="")
    parser.add_argument("--model_config", type=str, default=None, help="")
    parser.add_argument(
        "--mix_precision",
        type=str,
        default="bf16",
        choices=["fp16", "bf16", "fp32"],
        help="precision for training",
    )
    parser.add_argument("--wavelet_loss", action="store_true", help="")
    parser.add_argument("--not_resume_discriminator", action="store_true", help="")
    parser.add_argument("--not_resume_optimizer", action="store_true", help="")
    parser.add_argument("--wavelet_weight", type=float, default=0.1, help="")
    # Discriminator Model
    parser.add_argument("--load_disc_from_checkpoint", type=str, default=None, help="")
    parser.add_argument(
        "--disc_cls",
        type=str,
        default="causalvideovae.model.losses.LPIPSWithDiscriminator2D",
        help="",
    )
    parser.add_argument("--disc_start", type=int, default=5, help="")
    parser.add_argument("--disc_weight", type=float, default=0.5, help="")
    parser.add_argument("--kl_weight", type=float, default=1e-06, help="")
    parser.add_argument("--perceptual_weight", type=float, default=1.0, help="")
    parser.add_argument("--loss_type", type=str, default="l1", help="")
    parser.add_argument("--logvar_init", type=float, default=0.0, help="")

    # Validation
    parser.add_argument("--eval_steps", type=int, default=1000, help="")
    parser.add_argument("--eval_video_path", type=str, default=None, help="")
    parser.add_argument("--eval_num_frames", type=int, default=17, help="")
    parser.add_argument("--eval_resolution", type=int, default=256, help="")
    parser.add_argument("--eval_sample_rate", type=int, default=1, help="")
    parser.add_argument("--eval_batch_size", type=int, default=8, help="")
    parser.add_argument("--eval_subset_size", type=int, default=100, help="")
    parser.add_argument("--eval_num_video_log", type=int, default=2, help="")
    parser.add_argument("--eval_lpips", action="store_true", help="")

    # Dataset
    parser.add_argument("--dataset_num_worker", type=int, default=4, help="")

    # EMA
    parser.add_argument("--ema", action="store_true", help="")
    parser.add_argument("--ema_decay", type=float, default=0.999, help="")

    args = parser.parse_args()

    set_random_seed(args.seed)
    train(args)

if __name__ == "__main__":
    main()
