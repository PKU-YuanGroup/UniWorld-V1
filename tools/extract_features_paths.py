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
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from datasets.img_latent_dataset import ImgLatentDataset

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


class ImageFolderWithPath(ImageFolder):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_len = 64

    def __getitem__(self, index):
        path = self.samples[index][0]
        path = '/'.join(path.split('/')[-2:])
        encoded_path = path.encode("utf-8")
        assert len(encoded_path) < self.max_len
        paths_tensor = torch.zeros((self.max_len, ), dtype=torch.uint8)
        paths_tensor[:len(encoded_path)] = torch.tensor(list(encoded_path), dtype=torch.uint8)

        paths_recovered = "".join(map(chr, paths_tensor.tolist())).rstrip("\x00")
        assert paths_recovered == path

        return super().__getitem__(index) + (paths_tensor, )
    

if __name__ == "__main__":
    # Setup data:
    data_path = "/data/OpenDataLab___ImageNet-1K/raw/ImageNet-1K/val"
    dataset = ImageFolderWithPath(data_path, transform=img_transform(p_hflip=0.0, img_size=256))
    loaders = DataLoader(
            dataset,
            batch_size=2,
            shuffle=False,
            num_workers=0,
            pin_memory=True,
            drop_last=False
        ) 
    for batch_idx, batch_data in enumerate(loaders):
        x = batch_data[0]
        y = batch_data[1]  # (N,)
        path = batch_data[2]  # (N,)
            
