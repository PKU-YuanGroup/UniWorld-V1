import argparse
from tqdm import tqdm
import torch
import sys
from torch.utils.data import DataLoader, Subset
import os
from mmvae.causalvideovae.model import *
from accelerate import Accelerator
import numpy as np
from torchvision.datasets import ImageFolder
from torchvision import transforms
from PIL import Image
from glob import glob
from PIL import Image
import lpips
import cv2
from torch.utils.data import Dataset
from einops import rearrange
from diffusers.models import AutoencoderKL

def total_params(model):
    total_params = sum(p.numel() for p in model.parameters())
    total_params_in_millions = total_params / 1e6
    return int(total_params_in_millions)


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


def calculate_ssim(image1, image2):
    ssims = []
    image1 = image1.cpu().float()
    image2 = image2.cpu().float()
    image1 = rearrange(image1, 'b c h w -> (b c) h w').numpy().astype(np.float64)
    image2 = rearrange(image2, 'b c h w -> (b c) h w').numpy().astype(np.float64)

    for img1, img2 in zip(image1, image2):
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2
        kernel = cv2.getGaussianKernel(11, 1.5)
        window = np.outer(kernel, kernel.transpose())
        mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
        mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
        mu1_sq = mu1 ** 2
        mu2_sq = mu2 ** 2
        mu1_mu2 = mu1 * mu2
        sigma1_sq = cv2.filter2D(img1 ** 2, -1, window)[5:-5, 5:-5] - mu1_sq
        sigma2_sq = cv2.filter2D(img2 ** 2, -1, window)[5:-5, 5:-5] - mu2_sq
        sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2
        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                                (sigma1_sq + sigma2_sq + C2))
        ssims.append(ssim_map.mean())
    return np.array(ssims).mean()

def calculate_psnr(video_recon, inputs, device=None):
    mse = torch.mean(torch.square(inputs - video_recon), dim=list(range(video_recon.ndim))[1:])
    psnr = 20 * torch.log10(1 / torch.sqrt(mse))
    psnr = psnr.mean().detach()
    if psnr == torch.inf:
        return 100
    return psnr.cpu().item()

class ImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (str): Path to the directory containing images.
            transform (callable, optional): Transform to be applied to each image.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = glob(f'{root_dir}/*jpg')

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, 0


def save_tensor_as_image(tensor, save_path):
    """
    Save a tensor with value range [-1, 1] as an image.
    
    Args:
        tensor (torch.Tensor): Input tensor with shape (C, H, W) and values in [-1, 1].
        save_path (str): Path to save the image.
    """
    # Check tensor shape
    assert tensor.dim() == 3, "Tensor must have 3 dimensions (C, H, W)."
    tensor = torch.clip(tensor, -1, 1)
    
    # Rescale from [-1, 1] to [0, 1]
    tensor = (tensor + 1) / 2.0
    
    # Convert tensor to PIL image
    to_pil = transforms.ToPILImage()
    image = to_pil(tensor.detach().cpu().float())
    
    # Save the image
    image.save(save_path)

@torch.no_grad()
def main(args: argparse.Namespace):
    accelerator = Accelerator()
    device = accelerator.device
    
    device = args.device
    batch_size = args.batch_size
    num_workers = args.num_workers
    subset_size = args.subset_size
    
    
    data_type = torch.bfloat16
    if args.output_save_dir is not None:
        os.makedirs(args.output_save_dir, exist_ok=True)

    # ---- Load Model ----
    lpips_model = lpips.LPIPS(net="alex", spatial=True)
    lpips_model.to(device)
    lpips_model.requires_grad_(False)
    lpips_model.eval()
    

    device = args.device
    if args.hf_model is None:
        model_cls = ModelRegistry.get_model(args.model_name)
        vae = model_cls.from_pretrained(args.from_pretrained)
    else:
        vae = AutoencoderKL.from_pretrained(args.hf_model)
    
    vae = vae.to(device).to(data_type)
    if args.enable_tiling:
        vae.enable_tiling()

    # ---- Prepare Dataset ----
    transform = transforms.Compose([
        transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, args.resolution)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
    ])
    dataset = ImageDataset(args.image_path, transform=transform)
    # dataset = ImageFolder(args.image_path, transform=transform)

    if subset_size:
        indices = range(subset_size)
        dataset = Subset(dataset, indices=indices)
        
    dataloader = DataLoader(
        dataset, batch_size=batch_size, pin_memory=False, num_workers=num_workers
    )
    dataloader = accelerator.prepare(dataloader)

    # ---- Inference ----
    psnr_results = []
    lpips_results = []
    ssim_results = []
    cnt = 0
    for batch in tqdm(dataloader, disable=not accelerator.is_local_main_process):
        x, labels = batch
        labels = labels.reshape(-1, 1)
        x = x.to(device=device, dtype=data_type)  # b c h w
        
        if args.hf_model is not None:
            z = vae.encode(x).latent_dist.sample()
        else:
            z, z_t = vae.encode(x=x, input_ids=labels).latent_dist.sample()

        if args.hf_model is not None:
            image_recon = vae.decode(z).sample
        else:
            image_recon, text_recon = vae.decode(z, z_t).sample
        if args.output_save_dir is not None:
            for i in image_recon:
                save_path = f'{args.output_save_dir}/rank{accelerator.process_index}_{cnt}.jpg'
                save_tensor_as_image(i, save_path)
                cnt += 1

        
        psnr_results.append(calculate_psnr((torch.clip(image_recon, -1, 1) + 1) / 2, (torch.clip(x, -1, 1) + 1) / 2))

        ssim_results.append(calculate_ssim((torch.clip(image_recon, -1, 1) + 1) / 2, (torch.clip(x, -1, 1) + 1) / 2))
        lpips_score = (
                        lpips_model.forward(torch.clip(x, -1, 1), torch.clip(image_recon, -1, 1))
                        .mean()
                        .detach()
                        .cpu()
                        .item()
                    )
        lpips_results.append(lpips_score)

    psnr_results = torch.tensor(psnr_results).to(device)
    psnr_results_list = accelerator.gather(psnr_results)

    lpips_results = torch.tensor(lpips_results).to(device)
    lpips_results_list = accelerator.gather(lpips_results)

    ssim_results = torch.tensor(ssim_results).to(device)
    ssim_results_list = accelerator.gather(ssim_results)

    if accelerator.process_index == 0:
        print(
            f'psnr: {psnr_results_list.mean().item()}\n'
            f'lpips: {lpips_results_list.mean().item()}\n'
            f'ssim: {ssim_results_list.mean().item()}\n'
            f'enc param: {total_params(vae.encoder)}M\n'
            f'dec param: {total_params(vae.decoder)}M\n'
            )
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path", type=str, default="")
    parser.add_argument("--from_pretrained", type=str, default="")
    parser.add_argument("--resolution", type=int, default=336)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--subset_size", type=int, default=None)
    parser.add_argument('--enable_tiling', action='store_true')
    parser.add_argument('--cls_data', action='store_true')
    parser.add_argument("--model_name", type=str, default=None, help="")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output_save_dir", type=str, default=None)
    parser.add_argument("--hf_model", type=str, default=None)

    args = parser.parse_args()
    main(args)
    
