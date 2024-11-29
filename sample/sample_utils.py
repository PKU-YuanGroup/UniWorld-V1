
from einops import rearrange
import time
import torch
import os
import torch.distributed as dist
from torchvision.utils import save_image
import imageio
import math
import argparse
import random
import numpy as np

try:
    import torch_npu
    from opensora.npu_config import npu_config
except:
    torch_npu = None
    npu_config = None
from models import FlowWorld
from diffusers.models import AutoencoderKL

from .pipeline_flowworld import FlowWorldPipeline
from .scheduler import FlowMatchEulerScheduler

# adapted from https://github.com/huggingface/accelerate/blob/main/src/accelerate/utils/random.py#L31
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


def get_scheduler(args):
    scheduler = FlowMatchEulerScheduler()
    return scheduler

def prepare_pipeline(args, dtype, device):
    
    weight_dtype = dtype

    vae = AutoencoderKL.from_pretrained(f"cache_dir/models--stabilityai--sd-vae-ft-ema/snapshots/f04b2c4b98319346dad8c65879f680b1997b204a")
    vae = vae.to(device=device, dtype=weight_dtype).eval()


    transformer_model = FlowWorld.from_pretrained(
        args.model_path, cache_dir=args.cache_dir, 
        # device_map=None, 
        torch_dtype=weight_dtype
        ).eval()
    
    scheduler = get_scheduler(args)
    pipeline = FlowWorldPipeline(
        vae=vae,
        scheduler=scheduler,
        transformer=transformer_model, 
    ).to(device)

    if args.save_memory:
        print('enable_model_cpu_offload AND enable_sequential_cpu_offload')
        pipeline.enable_model_cpu_offload()
        pipeline.enable_sequential_cpu_offload()
        
    if args.compile:
        pipeline.transformer = torch.compile(pipeline.transformer)

    return pipeline

def init_gpu_env(args):
    local_rank = int(os.getenv('RANK', 0))
    world_size = int(os.getenv('WORLD_SIZE', 1))
    args.local_rank = local_rank
    args.world_size = world_size
    torch.cuda.set_device(local_rank)
    dist.init_process_group(
        backend='nccl', init_method='env://', 
        world_size=world_size, rank=local_rank
        )
    return args

def init_npu_env(args):
    local_rank = int(os.getenv('RANK', 0))
    world_size = int(os.getenv('WORLD_SIZE', 1))
    args.local_rank = local_rank
    args.world_size = world_size
    torch_npu.npu.set_device(local_rank)
    dist.init_process_group(
        backend='hccl', init_method='env://', 
        world_size=world_size, rank=local_rank
        )
    return args


def save_video_grid(video, nrow=None):
    b, t, h, w, c = video.shape

    if nrow is None:
        nrow = math.ceil(math.sqrt(b))
    ncol = math.ceil(b / nrow)
    padding = 1
    video_grid = torch.zeros(
        (
            t, 
            (padding + h) * nrow + padding, 
            (padding + w) * ncol + padding, 
            c
        ), 
        dtype=torch.uint8
        )

    for i in range(b):
        r = i // ncol
        c = i % ncol
        start_r = (padding + h) * r
        start_c = (padding + w) * c
        video_grid[:, start_r:start_r + h, start_c:start_c + w] = video[i]

    return video_grid


def run_model_and_save_samples(args, pipeline):
    dtype = torch.float32 if args.fp32 else torch.bfloat16

    if args.seed is not None:
        set_seed(args.seed, rank=args.local_rank, device_specific=True)
    if args.local_rank >= 0:
        torch.manual_seed(args.seed + args.local_rank)
    if not os.path.exists(args.save_img_path):
        os.makedirs(args.save_img_path, exist_ok=True)

    video_grids = []
    if not isinstance(args.label, list):
        args.label = [int(args.label)]
    if len(args.label) == 1 and args.label[0].endswith('txt'):
        label = open(args.label[0], 'r').readlines()
        args.label = [int(i.strip()) for i in label]
    
    
    def generate(label):
        videos = pipeline(
            label, 
            num_frames=args.num_frames,
            height=args.height,
            width=args.width,
            num_inference_steps=args.num_sampling_steps,
            guidance_scale=args.guidance_scale,
            num_samples_per_label=args.num_samples_per_label,
            use_linear_quadratic_schedule=args.use_linear_quadratic_schedule, 
        ).videos
        videos = rearrange(videos, 'b t h w c -> (b t) c h w')
        if args.num_samples_per_label != 1:
            for i, image in enumerate(videos):
                save_image(
                    image / 255.0, 
                    os.path.join(
                        args.save_img_path, 
                        f'{index}_gs{args.guidance_scale}_s{args.num_sampling_steps}_i{i}.jpg'
                        ),
                    nrow=math.ceil(math.sqrt(videos.shape[0])), 
                    normalize=True, 
                    value_range=(0, 1)
                    )  # b c h w
        save_image(
            videos / 255.0, 
            os.path.join(
                args.save_img_path, 
                f'{index}_gs{args.guidance_scale}_s{args.num_sampling_steps}.jpg'
                ),
            nrow=math.ceil(math.sqrt(videos.shape[0])), 
            normalize=True, 
            value_range=(0, 1)
            )  # b c h w
        video_grids.append(videos)

    for index, label in enumerate(args.label):
        if args.local_rank != -1 and index % args.world_size != args.local_rank:
            continue  # skip when ddp
        generate(label)

    if args.local_rank != -1:
        dist.barrier()
        assert len(args.label) >= args.world_size
        video_grids = torch.cat(video_grids, dim=0).cuda()  # num c h w or 1 t h w c
        
        shape = list(video_grids.shape)  # num c h w or 1 t h w c
        if args.num_frames == 1:
            max_sample = math.ceil(len(args.label) / args.world_size) * args.num_samples_per_label  # max = 8
        else:
            max_sample = math.ceil(len(args.label) / args.world_size)
        video_grids_to_gather = [torch.zeros(*shape[1:], dtype=torch.uint8).cuda() for _ in range(max_sample)]
        # true video are filled to video_grids_to_gather, maybe the last element is all zero.
        for i, v in enumerate(video_grids):
            video_grids_to_gather[i] = v
        video_grids_to_gather = torch.stack(video_grids_to_gather, dim=0)

        shape[0] = max_sample * args.world_size
        gathered_tensor = torch.zeros(shape, dtype=torch.uint8).cuda()
        dist.all_gather_into_tensor(gathered_tensor, video_grids_to_gather.contiguous())
        video_grids = gathered_tensor.cpu()
        which_to_save = torch.sum(video_grids, dim=list(range(video_grids.ndim))[1:]).bool()
        video_grids = video_grids[which_to_save]
        dist.barrier()
    else:
        video_grids = torch.cat(video_grids, dim=0)
            
    
    if args.local_rank <= 0:
        save_image(
            video_grids / 255.0, 
            os.path.join(
                args.save_img_path,
                f'gs{args.guidance_scale}_s{args.num_sampling_steps}.jpg'
                ), 
            nrow=math.ceil(math.sqrt(len(video_grids))), 
            normalize=True, 
            value_range=(0, 1)
            )
        print('save path {}'.format(args.save_img_path))



def run_model_and_save_samples_npu(args, pipeline, caption_refiner_model=None, enhance_video_model=None):
    
    # experimental_config = torch_npu.profiler._ExperimentalConfig(
    #     profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
    #     aic_metrics=torch_npu.profiler.AiCMetrics.PipeUtilization
    # )
    # profile_output_path = "/home/image_data/npu_profiling_t2v"
    # os.makedirs(profile_output_path, exist_ok=True)
    # with torch_npu.profiler.profile(
    #         activities=[
    #             torch_npu.profiler.ProfilerActivity.NPU, 
    #             torch_npu.profiler.ProfilerActivity.CPU
    #             ],
    #         with_stack=True,
    #         record_shapes=True,
    #         profile_memory=True,
    #         experimental_config=experimental_config,
    #         schedule=torch_npu.profiler.schedule(
    #             wait=10000, warmup=0, active=1, repeat=1, skip_first=0
    #             ),
    #         on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(f"{profile_output_path}/")
    # ) as prof:
    run_model_and_save_samples(args, pipeline, caption_refiner_model, enhance_video_model)
        # prof.step()


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default='LanguageBind/Open-Sora-Plan-v1.0.0')
    parser.add_argument("--num_frames", type=int, default=1)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--device", type=str, default='cuda:0')
    parser.add_argument("--cache_dir", type=str, default='./cache_dir')
    parser.add_argument("--save_img_path", type=str, default="./sample_videos/t2v")
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    parser.add_argument("--num_sampling_steps", type=int, default=50)
    parser.add_argument("--label", nargs='+')
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_samples_per_label", type=int, default=1)
    parser.add_argument('--compile', action='store_true')
    parser.add_argument('--save_memory', action='store_true') 
    parser.add_argument('--local_rank', type=int, default=-1)    
    parser.add_argument('--world_size', type=int, default=1)    
    parser.add_argument('--fp32', action='store_true')
    parser.add_argument('--use_linear_quadratic_schedule', action='store_true')
    args = parser.parse_args()
    return args