# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# LightningDiT: https://github.com/hustvl/LightningDiT
# --------------------------------------------------------

import os
import numpy as np
from glob import glob
from tqdm import tqdm
import torch
from torch.utils.data import Dataset
from safetensors import safe_open
from PIL import Image


class ImgLatentDataset(Dataset):
    def __init__(self, data_dir, latent_norm=True, latent_multiplier=1.0, raw_data_dir=None, raw_img_transform=None):
        self.data_dir = data_dir
        self.latent_norm = latent_norm
        self.latent_multiplier = latent_multiplier
        self.raw_data_dir = raw_data_dir
        self.raw_img_transform = raw_img_transform

        self.files = sorted(glob(os.path.join(data_dir, "*.safetensors")))
        self.img_to_file_map = self.get_img_to_safefile_map()
        
        if latent_norm:
            self._latent_mean, self._latent_std = self.get_latent_stats()

    def get_img_to_safefile_map(self):
        img_to_file = {}
        for safe_file in self.files:
            with safe_open(safe_file, framework="pt", device="cpu") as f:
                labels = f.get_slice('labels')
                labels_shape = labels.get_shape()
                num_imgs = labels_shape[0]
                cur_len = len(img_to_file)
                for i in range(num_imgs):
                    img_to_file[cur_len+i] = {
                        'safe_file': safe_file,
                        'idx_in_file': i
                    }
        return img_to_file

    def get_latent_stats(self):
        latent_stats_cache_file = os.path.join(self.data_dir, "latents_stats.pt")
        if not os.path.exists(latent_stats_cache_file):
            latent_stats = self.compute_latent_stats()
            torch.save(latent_stats, latent_stats_cache_file)
        else:
            latent_stats = torch.load(latent_stats_cache_file)
        return latent_stats['mean'], latent_stats['std']
    
    def compute_latent_stats(self):
        num_samples = min(10000, len(self.img_to_file_map))
        random_indices = np.random.choice(len(self.img_to_file_map), num_samples, replace=False)
        latents = []
        for idx in tqdm(random_indices):
            img_info = self.img_to_file_map[idx]
            safe_file, img_idx = img_info['safe_file'], img_info['idx_in_file']
            with safe_open(safe_file, framework="pt", device="cpu") as f:
                features = f.get_slice('latents')
                feature = features[img_idx:img_idx+1]
                latents.append(feature)
        latents = torch.cat(latents, dim=0)
        mean = latents.mean(dim=[0, 2, 3], keepdim=True)
        std = latents.std(dim=[0, 2, 3], keepdim=True)
        latent_stats = {'mean': mean, 'std': std}
        print(latent_stats)
        return latent_stats

    def __len__(self):
        return len(self.img_to_file_map.keys())

    def __getitem__(self, idx):
        img_info = self.img_to_file_map[idx]
        safe_file, img_idx = img_info['safe_file'], img_info['idx_in_file']
        with safe_open(safe_file, framework="pt", device="cpu") as f:
            no_flip = np.random.uniform(0, 1) > 0.5
            tensor_key = "latents" if no_flip else "latents_flip"
            features = f.get_slice(tensor_key)
            labels = f.get_slice('labels')
            feature = features[img_idx:img_idx+1]
            label = labels[img_idx:img_idx+1]
            if 'paths' in f.keys() and (self.raw_img_transform is not None) and (self.raw_data_dir is not None):
                paths = f.get_slice('paths')
                path_tensor = paths[img_idx:img_idx+1].squeeze(0)
                path = "".join(map(chr, path_tensor.tolist())).rstrip("\x00")
                path = os.path.join(self.raw_data_dir, path)
                assert os.path.exists(path)
                with open(path, "rb") as f:
                    image = Image.open(f).convert("RGB")
                image_tensor = self.raw_img_transform(image)
            else:
                image_tensor = torch.tensor(0)

        if self.latent_norm:
            feature = (feature - self._latent_mean) / self._latent_std
        feature = feature * self.latent_multiplier
        
        # remove the first batch dimension (=1) kept by get_slice()
        feature = feature.squeeze(0)
        label = label.squeeze(0)
        return feature, label, image_tensor
    


if __name__ == "__main__":
    from diffusers import AutoencoderKL
    from torchvision.utils import save_image
    import sys
    from torchvision import transforms
    from torchvision.datasets import ImageFolder
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    from tokenizer import VAE_Models
    from tools.extract_features import center_crop_arr


    crop_size = 256
    transform = transforms.Compose([
        transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, crop_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
    ])
    
    data_dir = '/data/checkpoints/LanguageBind/offline_feature/offline_sdvae_256_path/imagenet_train_256'
    vae_type, vae_path = 'sdvae', "/data/checkpoints/stabilityai/sd-vae-ft-ema/vae-ft-ema-560000-ema-pruned.safetensors"
    # data_dir = '/data/checkpoints/LanguageBind/offline_feature/offline_vavae_256_path/imagenet_train_256'
    # vae_type, vae_path = 'vavae', "/data/checkpoints/hustvl/vavae-imagenet256-f16d32-dinov2/vavae-imagenet256-f16d32-dinov2.pt"
    raw_data_dir = None
    dataset = ImgLatentDataset(data_dir, latent_norm=False, raw_data_dir=raw_data_dir, raw_img_transform=transform)
    print(dataset.get_latent_stats())
    num_sample_to_vis = 4
    samples = torch.stack([dataset.__getitem__(i)[0] for i in range(num_sample_to_vis)]).cuda()
    vae = VAE_Models[vae_type](vae_path)
    samples = vae.model.decode(samples)
    save_image(samples, "sdvae.png", nrow=4, normalize=True, value_range=(-1, 1))
    # save_image(samples, "vavae.png", nrow=4, normalize=True, value_range=(-1, 1))

    # for raw_data_dir is not None
    # samples = torch.stack([dataset.__getitem__(i)[2] for i in range(num_sample_to_vis)]).cuda()
    # samples = vae.encode_images(samples)
    # samples = vae.model.decode(samples)
    # save_image(samples, "sdvae_enc2dec.png", nrow=4, normalize=True, value_range=(-1, 1))
    # save_image(samples, "vavae_enc2dec.png", nrow=4, normalize=True, value_range=(-1, 1))