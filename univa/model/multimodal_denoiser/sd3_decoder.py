import torch
import torch.nn as nn

from transformers import CLIPVisionModel, CLIPImageProcessor, CLIPVisionConfig
from diffusers import StableDiffusion3Pipeline
from .scheduler import OpenSoraPlanFlowMatchEulerScheduler

class SD3DenoiseTower(nn.Module):
    def __init__(self, denoise_tower, args, delay_load=False):
        super().__init__()

        self.is_loaded = False
        self.noise_scheduler = OpenSoraPlanFlowMatchEulerScheduler()
        self.denoise_tower_name = denoise_tower
        self.unfreeze = getattr(args, 'unfreeze_mm_denoise_tower', False)

        assert not delay_load
        if not delay_load:
            self.load_model()

    def load_model(self, device_map=None, pretrained=None):
        if self.is_loaded:
            print('{} is already loaded, `load_model` called again, skipping.'.format(self.vision_tower_name))
            return

        self.denoise_tower = StableDiffusion3Pipeline.from_pretrained(self.denoise_tower_name, device_map=device_map)
        print(f'[debug]\tload pretrained weight from {self.denoise_tower_name}')
        print(f'[debug]\tis training? {self.unfreeze}')
        self.denoise_tower.requires_grad_(self.unfreeze)
        print(f"[debug]\tself.denoise_tower.requires_grad="
              f"{self.denoise_tower.transformer.pos_embed.proj.weight.requires_grad}")

        if pretrained is not None:
            print(f"=> loading pretrained mm_denoise_tower from {self.pretrained} ...")
            self.denoise_tower.load_state_dict(torch.load(self.pretrained, map_location='cpu'))

        self.is_loaded = True

    def inner_forward_batch(self, image, encoder_hidden_state):
        bsz, _, height, width = image.shape
        image = self.denoise_tower.image_processor.preprocess(image, height=height, width=width)
        image = image.to(device=self.device, dtype=self.dtype)
        latent = self.denoise_tower.vae.encode(image).latent_dist.sample()
        model_input = (latent / self.denoise_tower.vae.config.scaling_factor) + self.denoise_tower.vae.config.shift_factor

        noise = torch.randn_like(model_input)
        target = noise - model_input

        # Sample a random timestep for each image
        # for weighting schemes where we sample timesteps non-uniformly
        sigmas = self.noise_scheduler.compute_density_for_sigma_sampling(
            batch_size=bsz,
            logit_mean=0.0,
            logit_std=1.0,
        ).to(device=self.device)
        timestep = sigmas.clone() * self.noise_scheduler.rescale  # rescale to [0, 1000.0)
        while sigmas.ndim < latent.ndim:
            sigmas = sigmas.unsqueeze(-1)

        noisy_model_input = self.noise_scheduler.add_noise(latent, sigmas, noise)
        model_pred = self.denoise_tower.transformer(
            hidden_states=noisy_model_input, 
            encoder_hidden_states=encoder_hidden_state, 
            timestep=timestep, 
        )
        return model_pred, target

    def inner_forward(
        self,
        images: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
    ):
        bsz = len(images)
        
        if isinstance(images, list):
            model_pred, target = tuple(zip(*[self.inner_forward_batch(self, images[i: i+1], encoder_hidden_states[i: i+1]) for i in range(bsz)]))
        else:
            model_pred, target = self.inner_forward_batch(self, images, encoder_hidden_states)
        return {'model_pred': model_pred, 'target': target}

    def forward(self, images):
        if self.unfreeze:
            denoiser_output = self.inner_forward(images)
            return denoiser_output
        else:
            with torch.no_grad():
                denoiser_output = self.inner_forward(images)
                return denoiser_output

    @torch.no_grad()
    def sample(self, z, temperature=1.0, cfg=1.0):
        pass

    @property
    def dtype(self):
        return self.denoise_tower.dtype

    @property
    def device(self):
        return self.denoise_tower.device

    @property
    def config(self):
        if self.is_loaded:
            return self.denoise_tower.config
        else:
            return self.cfg_only

    @property
    def hidden_size(self):
        return self.config.hidden_size

    # @property
    def image_size(self):
        return self.config.image_size
    
    # @property
    def patch_size(self):
        return self.config.patch_size
    
    @property
    def num_patches_per_side(self):
        return self.config.image_size // self.config.patch_size

    @property
    def num_patches(self):
        return (self.config.image_size // self.config.patch_size) ** 2


