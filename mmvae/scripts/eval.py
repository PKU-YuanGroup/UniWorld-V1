import os
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import sys
from glob import glob
import multiprocessing as mp
sys.path.append(".")
from causalvideovae.eval.cal_lpips import calculate_lpips
from causalvideovae.eval.cal_fvd import calculate_fvd
from causalvideovae.eval.cal_psnr import calculate_psnr
from causalvideovae.eval.cal_ssim import calculate_ssim
from causalvideovae.dataset.video_dataset import (
    ValidVideoDataset,
    DecordInit,
    Compose,
    Lambda,
    resize,
    CenterCropVideo,
    ToTensorVideo
)
from accelerate import Accelerator

class EvalDataset(ValidVideoDataset):
    def __init__(
        self,
        real_video_dir,
        generated_video_dir,
        num_frames,
        sample_rate=1,
        crop_size=None,
        resolution=128,
    ) -> None:
        self.is_main_process = False
        self.v_decoder = DecordInit()
        self.real_video_files = []
        self.generated_video_files = self._make_dataset(generated_video_dir)
        for video_file in self.generated_video_files:
            filename = os.path.basename(video_file)
            if not os.path.exists(os.path.join(real_video_dir, filename)):
                raise Exception(os.path.join(real_video_dir, filename))
            self.real_video_files.append(os.path.join(real_video_dir, filename))
        self.num_frames = num_frames
        self.sample_rate = sample_rate
        self.crop_size = crop_size
        self.short_size = resolution
        self.transform = Compose(
            [
                ToTensorVideo(),
                Lambda(lambda x: resize(x, self.short_size)),
                (
                    CenterCropVideo(crop_size)
                    if crop_size is not None
                    else Lambda(lambda x: x)
                ),
            ]
        )

    def _make_dataset(self, real_video_dir):
        samples = []
        samples += sum(
            [
                glob(os.path.join(real_video_dir, f"*.{ext}"), recursive=True)
                for ext in self.video_exts
            ],
            [],
        )
        return samples
    
    def __len__(self):
        return len(self.real_video_files)

    def __getitem__(self, index):
        if index >= len(self):
            raise IndexError(len(self.generated_video_files))
        real_video_file = self.real_video_files[index]
        generated_video_file = self.generated_video_files[index]
        real_video_tensor = self._load_video(real_video_file, self.sample_rate)
        generated_video_tensor = self._load_video(generated_video_file, 1)
        return {"real": self.transform(real_video_tensor), "generated": self.transform(generated_video_tensor)}


def calculate_common_metric(accelerator, args, dataloader, device):
    score_list = []
    if args.metric == "fvd":
        if args.fvd_method == 'styleganv':
            from causalvideovae.eval.fvd.styleganv.fvd import load_i3d_pretrained
        elif args.fvd_method == 'videogpt':
            from causalvideovae.eval.fvd.videogpt.fvd import load_i3d_pretrained
        print(args.fvd_method)
        i3d = load_i3d_pretrained(device)
        
    for batch_data in tqdm(dataloader, disable=not accelerator.is_local_main_process):
        real_videos = batch_data["real"].to(device)
        generated_videos = batch_data["generated"].to(device)

        assert real_videos.shape[2] == generated_videos.shape[2]
        if args.metric == "fvd":
            tmp_list = list(
                calculate_fvd(
                    real_videos, generated_videos, args.device, i3d=i3d, method=args.fvd_method
                )["value"].values()
            )
        elif args.metric == "ssim":
            tmp_list = list(
                calculate_ssim(real_videos, generated_videos)["value"].values()
            )
        elif args.metric == "psnr":
            tmp_list = [calculate_psnr(real_videos, generated_videos)]
        else:
            tmp_list = [calculate_lpips(real_videos, generated_videos, args.device)]
        score_list += tmp_list
    score_list = [np.mean(score_list)]
    score_list = accelerator.gather_for_metrics(score_list)
    if accelerator.is_main_process:
        return np.mean(score_list)
    return None

def calculate_common_metric_mp(args, dataloader):
    pool = mp.Pool(processes=mp.cpu_count())
    results = []
    for data in tqdm(dataloader, desc="submit task to process"):
        real_videos = data["real"]
        generated_videos = data["generated"]
        assert real_videos.shape == generated_videos.shape
        if args.metric == "fvd":
            raise RuntimeError("Use cuda")
        elif args.metric == "ssim":
            results.append(pool.apply_async(calculate_ssim, args=(real_videos, generated_videos)))
        elif args.metric == "psnr":
            results.append(pool.apply_async(calculate_psnr, args=(real_videos, generated_videos)))
        elif args.metric == "lpips":
            ...
        else:
            raise NotImplementedError("No metric")
    
    
    score_list = []
    for result in tqdm(results):
        score_list += list(result.get()["value"].values())
        
    pool.close()
    pool.join()
    
    return np.mean(score_list)
        

def main():
    accelerator = Accelerator()
    
    if args.device is None:
        device = torch.device("cuda" if (torch.cuda.is_available()) else "cpu")
    else:
        device = torch.device(args.device)

    if args.num_workers is None:
        try:
            num_cpus = len(os.sched_getaffinity(0))
        except AttributeError:
            num_cpus = os.cpu_count()
        num_workers = min(num_cpus, 8) if num_cpus is not None else 0
    else:
        num_workers = args.num_workers

    dataset = EvalDataset(
        args.real_video_dir,
        args.generated_video_dir,
        num_frames=args.num_frames,
        sample_rate=args.sample_rate,
        crop_size=args.crop_size,
        resolution=args.resolution,
    )
    
    if args.subset_size:
        indices = list(range(args.subset_size))
        dataset = Subset(dataset, indices=indices)

    dataloader = DataLoader(
        dataset, args.batch_size, num_workers=num_workers, pin_memory=True
    )
    if args.mp:
        if not accelerator.is_main_process:
            raise Exception("MP mode shouldn't use accelerate")
        metric_score = calculate_common_metric_mp(args, dataloader)
        print(metric_score)
    else:
        dataloader = accelerator.prepare(dataloader)
        metric_score = calculate_common_metric(accelerator, args, dataloader, device)
        if accelerator.is_main_process:
            print(metric_score)


def parse_args():
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size to use")
    parser.add_argument("--real_video_dir", type=str, help=("the path of real videos`"))
    parser.add_argument("--fvd_method", type=str, default="styleganv")
    parser.add_argument(
        "--generated_video_dir", type=str, help=("the path of generated videos`")
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use. Like cuda, cuda:0 or cpu",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=8,
        help=(
            "Number of processes to use for data loading. "
            "Defaults to `min(8, num_cpus)`"
        ),
    )
    parser.add_argument("--sample_fps", type=int, default=30)
    parser.add_argument("--resolution", type=int, default=336)
    parser.add_argument("--crop_size", type=int, default=None)
    parser.add_argument("--num_frames", type=int, default=100)
    parser.add_argument("--sample_rate", type=int, default=1)
    parser.add_argument("--subset_size", type=int, default=None)
    parser.add_argument("--mp", action="store_true", help="")
    parser.add_argument(
        "--metric",
        type=str,
        default="fvd",
        choices=["fvd", "psnr", "ssim", "lpips", "flolpips"],
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main()
