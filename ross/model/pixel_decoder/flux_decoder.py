import torch.nn as nn
from diffusers import AutoencoderKL


class FluxDecoder(nn.Module):
    def __init__(self, config, **kwargs):
        super().__init__()
        self.config = config

        self.pixel_decoder = AutoencoderKL.from_pretrained(config.mm_pixel_decoder)
        self.pixel_decoder.requires_grad_(False)
        self.pixel_decoder.float()
        self.pixel_decoder.eval()

    @property
    def scaling_factor(self):
        return self.pixel_decoder.config.scaling_factor

    @property
    def shift_factor(self):
        return self.pixel_decoder.config.shift_factor

    @property
    def latent_dim(self):
        return self.pixel_decoder.config.latent_channels * 4

    def encode(self, x):
        return self.pixel_decoder.encode(x)

    def decode(self, z):
        return self.pixel_decoder.decode(z)
