"""
Vision Foundation Model Aligned VAE.
It has exactly the same architecture as the LDM VAE (or VQGAN-KL).
Here we first provide its inference implementation with diffusers. 
The training code will be provided later. 

"LightningDiT + VA_VAE" achieves state-of-the-art Latent Diffusion System
with 0.27 rFID and 1.35 FID on ImageNet 256x256.

by Maple (Jingfeng Yao) from HUST-VL
"""

import torch
import numpy as np
from PIL import Image
from omegaconf import OmegaConf
from torchvision import transforms
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from tokenizer.autoencoder import AutoencoderKL

class VA_VAE:
    """Vision Foundation Model Aligned VAE Implementation"""
    
    def __init__(self, ckpt_path, img_size=256, horizon_flip=0.5, fp16=True):
        """Initialize VA_VAE
        Args:
            config: Configuration dict containing img_size, horizon_flip and fp16 parameters
        """
        self.embed_dim = 32
        self.ckpt_path = ckpt_path
        self.img_size = img_size
        self.horizon_flip = horizon_flip
        self.load()

    def load(self):
        """Load and initialize VAE model"""
        self.model = AutoencoderKL(
            embed_dim=self.embed_dim,
            ch_mult=(1, 1, 2, 2, 4),
            ckpt_path=self.ckpt_path
        ).cuda().eval()
        return self
    
    def img_transform(self, p_hflip=0, img_size=None):
        """Image preprocessing transforms
        Args:
            p_hflip: Probability of horizontal flip
            img_size: Target image size, use default if None
        Returns:
            transforms.Compose: Image transform pipeline
        """
        img_size = img_size if img_size is not None else self.img_size
        img_transforms = [
            transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, img_size)),
            transforms.RandomHorizontalFlip(p=p_hflip),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
        ]
        return transforms.Compose(img_transforms)

    def encode_images(self, images):
        """Encode images to latent representations
        Args:
            images: Input image tensor
        Returns:
            torch.Tensor: Encoded latent representation
        """
        with torch.no_grad():
            posterior = self.model.encode(images.cuda())
            return posterior.sample()

    def decode_to_images(self, z):
        """Decode latent representations to images
        Args:
            z: Latent representation tensor
        Returns:
            np.ndarray: Decoded image array
        """
        with torch.no_grad():
            images = self.model.decode(z.cuda())
            images = torch.clamp(127.5 * images + 128.0, 0, 255).permute(0, 2, 3, 1).to("cpu", dtype=torch.uint8).numpy()
        return images

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


if __name__ == "__main__":
    vae = VA_VAE('/data/checkpoints/hustvl/vavae-imagenet256-f16d32-dinov2/vavae-imagenet256-f16d32-dinov2.pt')
    vae.load()
    print(vae.model)
    print(f"VA_VAE Parameters: {sum(p.numel() for p in vae.model.parameters()) / 1e6:.2f}M")
    print(f"VA_VAE Encoder Parameters: {sum(p.numel() for p in vae.model.encoder.parameters()) / 1e6:.2f}M")
    print(f"VA_VAE Decoder Parameters: {sum(p.numel() for p in vae.model.decoder.parameters()) / 1e6:.2f}M")
    image_path = '/data/OpenDataLab___ImageNet-1K/sample/image/sample_0440.jpg'
    transform = vae.img_transform()
    x = transform(Image.open(image_path)).unsqueeze(0).cuda()
    print(x.shape)
    out = vae.decode_to_images(vae.encode_images(x))[0]
    Image.fromarray(out).save('vavae.jpg')