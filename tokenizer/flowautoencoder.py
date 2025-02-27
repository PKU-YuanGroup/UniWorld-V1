import os
import torch
import math
from tqdm import tqdm
import numpy as np
import torch.nn as nn
from PIL import Image
from timm.models.vision_transformer import PatchEmbed, Mlp
from einops import rearrange
import torch.nn.functional as F


import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), ".")))
from autoencoder import DiagonalGaussianDistribution, Encoder, Decoder


class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    Same as DiT.
    """
    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256) -> None:
        super().__init__()
        self.frequency_embedding_size = frequency_embedding_size
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )

    @staticmethod
    def timestep_embedding(t: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
        """
        Create sinusoidal timestep embeddings.
        Args:
            t: A 1-D Tensor of N indices, one per batch element. These may be fractional.
            dim: The dimension of the output.
            max_period: Controls the minimum frequency of the embeddings.
        Returns:
            An (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
            
        return embedding
    
    def forward(self, t: torch.Tensor) -> torch.Tensor:
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb
    
def nonlinearity(x):
    # # swish
    # return x * torch.sigmoid(x)
    # swish
    return F.silu(x)


class GroupNorm(nn.Module):
    def __init__(self, num_channels, num_groups=32, eps=1e-6, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.norm = torch.nn.GroupNorm(
            num_groups=num_groups, num_channels=num_channels, eps=eps, affine=True
        )
    def forward(self, x):
        return self.norm(x)
    

class RMSNorm(torch.nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-6, *args, **kwargs):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(num_channels))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        x = rearrange(x, "b c h w -> b h w c")
        output = self._norm(x.float()).type_as(x)
        x = output * self.weight
        x = rearrange(x, "b h w c -> b c h w")
        return x
    
class LayerNorm(nn.Module):
    def __init__(self, num_channels, eps=1e-6, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.norm = torch.nn.LayerNorm(num_channels, eps=eps, elementwise_affine=True)
    def forward(self, x):
        x = rearrange(x, "b c h w -> b h w c")
        x = self.norm(x)
        x = rearrange(x, "b h w c -> b c h w")
        return x

def Normalize(in_channels, num_groups=32, norm_type="groupnorm"):
    if norm_type == "groupnorm":
        return torch.nn.GroupNorm(
            num_groups=num_groups, num_channels=in_channels, eps=1e-6, affine=True
        )
    elif norm_type == "layernorm":
        return LayerNorm(num_channels=in_channels, eps=1e-6)
    elif norm_type == "rmsnorm":
        return RMSNorm(num_channels=in_channels, eps=1e-6)

class Upsample(nn.Module):
    def __init__(self, in_channels, with_conv, upsampler='nearest'):
        super().__init__()
        self.with_conv = with_conv
        self.upsampler = upsampler
        self.antialias = False if upsampler == 'nearest' else True
        if self.with_conv:
            self.conv = torch.nn.Conv2d(
                in_channels, in_channels, kernel_size=3, stride=1, padding=1
            )

    def forward(self, x):
        x = torch.nn.functional.interpolate(x, scale_factor=2.0, mode=self.upsampler, antialias=self.antialias)
        if self.with_conv:
            x = self.conv(x)
        return x

class ResnetBlock(nn.Module):
    def __init__(
        self,
        *,
        in_channels,
        out_channels=None,
        conv_shortcut=False,
        dropout,
        temb_channels=512,
        norm_type='groupnorm', 
    ):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut

        self.norm1 = Normalize(in_channels, norm_type=norm_type)
        self.conv1 = torch.nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=1, padding=1
        )
        if temb_channels > 0:
            self.temb_proj = torch.nn.Conv2d(temb_channels, out_channels, 1)
        self.norm2 = Normalize(out_channels, norm_type=norm_type)
        self.dropout = torch.nn.Dropout(dropout)
        self.conv2 = torch.nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1
        )
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = torch.nn.Conv2d(
                    in_channels, out_channels, kernel_size=3, stride=1, padding=1
                )
            else:
                self.nin_shortcut = torch.nn.Conv2d(
                    in_channels, out_channels, kernel_size=1, stride=1, padding=0
                )

    def forward(self, x, temb):
        h = x
        h = self.norm1(h)
        h = nonlinearity(h)
        h = self.conv1(h)

        if temb is not None:
            h = h + self.temb_proj(nonlinearity(temb))

        h = self.norm2(h)
        h = nonlinearity(h)
        h = self.dropout(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)
            else:
                x = self.nin_shortcut(x)

        return x + h


class AttnBlock(nn.Module):
    def __init__(self, in_channels, norm_type='groupnorm'):
        super().__init__()
        self.in_channels = in_channels

        self.norm = Normalize(in_channels, norm_type=norm_type)
        self.q = torch.nn.Conv2d(
            in_channels, in_channels, kernel_size=1, stride=1, padding=0
        )
        self.k = torch.nn.Conv2d(
            in_channels, in_channels, kernel_size=1, stride=1, padding=0
        )
        self.v = torch.nn.Conv2d(
            in_channels, in_channels, kernel_size=1, stride=1, padding=0
        )
        self.proj_out = torch.nn.Conv2d(
            in_channels, in_channels, kernel_size=1, stride=1, padding=0
        )

    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # # compute attention
        # b, c, h, w = q.shape
        # q = q.reshape(b, c, h * w)
        # q = q.permute(0, 2, 1)  # b,hw,c
        # k = k.reshape(b, c, h * w)  # b,c,hw
        # w_ = torch.bmm(q, k)  # b,hw,hw    w[b,i,j]=sum_c q[b,i,c]k[b,c,j]
        # w_ = w_ * (int(c) ** (-0.5))
        # w_ = torch.nn.functional.softmax(w_, dim=2)

        # # attend to values
        # v = v.reshape(b, c, h * w)
        # w_ = w_.permute(0, 2, 1)  # b,hw,hw (first hw of k, second of q)
        # h_ = torch.bmm(v, w_)  # b, c,hw (hw of q) h_[b,c,j] = sum_i v[b,c,i] w_[b,i,j]
        # h_ = h_.reshape(b, c, h, w)

        b, c, h, w = q.shape
        q = rearrange(q, 'b c h w -> b 1 (h w) c')
        k = rearrange(k, 'b c h w -> b 1 (h w) c')
        v = rearrange(v, 'b c h w -> b 1 (h w) c')

        attn_output = F.scaled_dot_product_attention(
            q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False
        )
        h_ = rearrange(attn_output, 'b 1 (h w) c -> b c h w', h=h, w=w)

        h_ = self.proj_out(h_)

        return x + h_


class FlowDecoder(nn.Module):
    def __init__(
        self,
        *,
        ch=128,
        out_ch=3,
        ch_mult=(1, 1, 2, 2, 4),
        num_res_blocks=2,
        attn_resolutions=(16,),
        dropout=0.0,
        resamp_with_conv=True,
        in_channels=3,
        resolution=256,
        z_channels=16,
        give_pre_end=False,
        temb_ch=512, 
        norm_type='groupnorm', 
        upsampler='nearest', 
        **ignore_kwargs,
    ):
        super().__init__()
        self.ch = ch
        self.temb_ch = temb_ch
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels
        self.give_pre_end = give_pre_end

        # compute in_ch_mult, block_in and curr_res at lowest res
        in_ch_mult = (1,) + tuple(ch_mult)
        block_in = ch * ch_mult[self.num_resolutions - 1]
        curr_res = resolution // 2 ** (self.num_resolutions - 1)
        self.z_shape = (1, z_channels, curr_res, curr_res)
        # print(
        #     "Working with z of shape {} = {} dimensions.".format(
        #         self.z_shape, np.prod(self.z_shape)
        #     )
        # )

        # z to block_in
        self.conv_in = torch.nn.Conv2d(
            z_channels, block_in, kernel_size=3, stride=1, padding=1
        )

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(
            in_channels=block_in,
            out_channels=block_in,
            temb_channels=self.temb_ch,
            dropout=dropout,
            norm_type=norm_type, 
            conv_shortcut=False
        )
        self.mid.attn_1 = AttnBlock(block_in, norm_type=norm_type)
        self.mid.block_2 = ResnetBlock(
            in_channels=block_in,
            out_channels=block_in,
            temb_channels=self.temb_ch,
            dropout=dropout,
            norm_type=norm_type, 
            conv_shortcut=False
        )

        # upsampling
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch * ch_mult[i_level]
            for i_block in range(self.num_res_blocks + 1):
                block.append(
                    ResnetBlock(
                        in_channels=block_in,
                        out_channels=block_out,
                        temb_channels=self.temb_ch,
                        dropout=dropout,
                        norm_type=norm_type, 
                    )
                )
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(AttnBlock(block_in, norm_type=norm_type))
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0:
                up.upsample = Upsample(block_in, resamp_with_conv, upsampler)
                curr_res = curr_res * 2
            self.up.insert(0, up)  # prepend to get consistent order

        # end
        self.norm_out = Normalize(block_in, norm_type=norm_type)
        self.conv_out = torch.nn.Conv2d(
            block_in, out_ch, kernel_size=3, stride=1, padding=1
        )

    def forward(self, z, temb):
        # assert z.shape[1:] == self.z_shape[1:]
        self.last_z_shape = z.shape

        # z to block_in
        h = self.conv_in(z)

        # middle
        temb_idx = 0
        h = self.mid.block_1(h, temb[temb_idx])
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb[temb_idx])

        # upsampling
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                h = self.up[i_level].block[i_block](h, temb[temb_idx])
                # print(torch.any(torch.isnan(h)), torch.max(h).item(), torch.min(h).item(), torch.mean(h).item(), torch.std(h).item())
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)
                    # print(torch.any(torch.isnan(h)), torch.max(h).item(), torch.min(h).item(), torch.mean(h).item(), torch.std(h).item())
            if i_level != 0:
                h = self.up[i_level].upsample(h)
                # print(torch.any(torch.isnan(h)), torch.max(h).item(), torch.min(h).item(), torch.mean(h).item(), torch.std(h).item())
                temb_idx += 1

        # end
        if self.give_pre_end:
            return h

        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        # print(torch.any(torch.isnan(h)), torch.max(h).item(), torch.min(h).item(), torch.mean(h).item(), torch.std(h).item())
        return h


class FlowAE(nn.Module):
    def __init__(
            self, 
            embed_dim, 
            ch_mult, 
            input_size=256,
            patch_size=8,
            ch=128, 
            use_variational=True, 
            ckpt_path=None, 
            model_type='vavae', 
            sample_mode=False, 
            norm_type='rmsnorm', 
            multi_latent=True, 
            add_y_to_x=False, 
            upsampler='nearest'
            ):
        super().__init__()
        self.ch_mult = ch_mult
        self.embed_dim = embed_dim
        self.model_type = model_type
        self.use_variational = use_variational
        mult = 2 if self.use_variational else 1
        hidden_size = ch * ch_mult[-1]

        # decode
        if model_type == 'vavae' or model_type == 'sdvae':
            self.flow = FlowDecoder(
                ch=ch, ch_mult=ch_mult, z_channels=ch * ch_mult[-1], 
                attn_resolutions=(16,), temb_ch=hidden_size, norm_type=norm_type, upsampler=upsampler
                )
        elif model_type == 'marvae':
            self.flow = FlowDecoder(
                ch=ch, ch_mult=ch_mult, z_channels=ch * ch_mult[-1], 
                attn_resolutions=(), temb_ch=hidden_size, norm_type=norm_type, upsampler=upsampler
                )

        # flow
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.add_y_to_x = add_y_to_x
        self.multi_latent = multi_latent
        if self.multi_latent:
            self.y_embedder = nn.ModuleList([
                torch.nn.Conv2d(embed_dim, hidden_size, 1) if i == 0 else Upsample(hidden_size, with_conv=True, upsampler=upsampler) 
                for i in range(len(ch_mult))
                ])
        else:
            self.y_embedder = torch.nn.Conv2d(embed_dim, hidden_size, 1)
        self.x_embedder = PatchEmbed(input_size, patch_size, 3, hidden_size, bias=True)
        num_patches = self.x_embedder.num_patches
        # Will use fixed sin-cos embedding:
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, hidden_size), requires_grad=False)
        self.initialize_weights()

        # init 
        if ckpt_path is not None and sample_mode:
            mult = 2 if self.use_variational else 1
            self.encoder = Encoder(ch_mult=ch_mult, z_channels=embed_dim)
            if model_type == 'vavae' or model_type == 'sdvae':
                self.decoder = Decoder(ch_mult=ch_mult, z_channels=embed_dim, attn_resolutions=(16,))
            elif model_type == 'marvae':
                self.decoder = Decoder(ch_mult=ch_mult, z_channels=embed_dim, attn_resolutions=())
            self.use_variational = use_variational
            mult = 2 if self.use_variational else 1
            self.quant_conv = torch.nn.Conv2d(2 * embed_dim, mult * embed_dim, 1)
            self.post_quant_conv = torch.nn.Conv2d(embed_dim, embed_dim, 1)
            self.init_from_ckpt(ckpt_path)

    def init_from_ckpt(self, path):
        if self.model_type == 'vavae':
            sd = torch.load(path, map_location="cpu")
            if 'state_dict' in sd.keys():
                sd = sd['state_dict']
            sd = {k: v for k, v in sd.items() if 'foundation_model.model' and 'loss' not in k}
            self.load_state_dict(sd, strict=False)
        elif self.model_type == 'sdvae':
            from safetensors.torch import load_file
            sd = load_file(path)
            self.load_state_dict(sd, strict=False)
        elif self.model_type == 'marvae':
            sd = torch.load(path, map_location="cpu")["model"]
            self.load_state_dict(sd, strict=False)
        else:
            raise ValueError(f"Invalid model type: {self.model_type}")
    
    def initialize_weights(self):

        # Initialize (and freeze) pos_embed by sin-cos embedding:
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.x_embedder.num_patches ** 0.5))
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

    def encode(self, x):
        h = self.encoder(x)
        moments = self.quant_conv(h)
        if not self.use_variational:
            moments = torch.cat((moments, torch.ones_like(moments)), 1)
        posterior = DiagonalGaussianDistribution(moments)
        return posterior

    def decode(self, z):
        z = self.post_quant_conv(z)
        dec = self.decoder(z)
        return dec

    @torch.compile
    def forward(self, x, t=None, y=None, **kwargs):
        x = self.x_embedder(x) + self.pos_embed
        h = w = int(x.shape[1] ** 0.5)
        x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w).contiguous()  # B C H W
        t = self.t_embedder(t)[:, :, None, None]    # (B, C) -> # (B, C, 1, 1)
        c = []
        c = [self.y_embedder(y) if i==0 else t for i in range(len(self.ch_mult))]

        x = self.flow(x, c) # B 3 256 256
        return x    
    
    def forward_with_cfg(self, x, t, y, cfg_scale, cfg_interval=None, cfg_interval_start=None):
        """
        Forward pass of DiT, but also batches the unconditional forward pass for classifier-free guidance.
        """
        # https://github.com/openai/glide-text2im/blob/main/notebooks/text2im.ipynb
        half = x[: len(x) // 2]
        combined = torch.cat([half, half], dim=0)
        model_out = self.forward(combined, t, y)
        # For exact reproducibility reasons, we apply classifier-free guidance on only
        # three channels by default. The standard approach to cfg applies it to all channels.
        # This can be done by uncommenting the following line and commenting-out the line following that.
        # eps, rest = model_out[:, :self.in_channels], model_out[:, self.in_channels:]
        eps, rest = model_out[:, :3], model_out[:, 3:]
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
        half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        
        if cfg_interval is True:
            timestep = t[0]
            if timestep < cfg_interval_start:
                half_eps = cond_eps

        eps = torch.cat([half_eps, half_eps], dim=0)
        return torch.cat([eps, rest], dim=1)



def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb

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


#################################################################################
#                             FlowVAE Configs                              #
#################################################################################

def FlowVAVAE(**kwargs):
    return FlowAE(embed_dim=32, patch_size=16, ch_mult=(1, 1, 2, 2, 4), model_type='vavae', **kwargs)

def FlowSDVAE(**kwargs):
    return FlowAE(embed_dim=4, patch_size=8, ch_mult=(1, 2, 4, 4), model_type='sdvae', **kwargs)

def FlowMARVAE(**kwargs):
    return FlowAE(embed_dim=16, patch_size=16, ch_mult=(1, 1, 2, 2, 4), model_type='marvae', **kwargs)

FlowVAE_models = {
    'FlowVAVAE':  FlowVAVAE,  'FlowSDVAE':  FlowSDVAE,  'FlowMARVAE':  FlowMARVAE,  
}


if __name__ == '__main__':
    import yaml
    import sys
    from torch.utils.data import DataLoader
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    from tools.extract_features import center_crop_arr
    from torchvision import transforms
    from datasets.img_latent_dataset import ImgLatentDataset
    from torchvision.utils import save_image
    from models.lpips import LPIPS
    from transport import create_transport, Sampler

    config_path = 'configs/flowsdvae_50kx512.yaml'
    with open(config_path, "r") as f:
        train_config = yaml.safe_load(f)
    input_size = train_config['data']['image_size']
    if 'downsample_ratio' in train_config['vae']:
        downsample_ratio = train_config['vae']['downsample_ratio']
    else:
        downsample_ratio = 8
    assert train_config['data']['image_size'] % downsample_ratio == 0, "Image size must be divisible by 8 (for the VAE encoder)."
    latent_size = train_config['data']['image_size'] // downsample_ratio
    in_channels = 3
    model = FlowVAE_models[train_config['vae']['vae_type']](
        input_size=train_config['data']['image_size'],
        norm_type=train_config['vae']['norm_type'] if 'norm_type' in train_config['vae'] else 'rmsnorm',
        multi_latent=train_config['vae']['multi_latent'] if 'multi_latent' in train_config['vae'] else True,
        add_y_to_x=train_config['vae']['add_y_to_x'] if 'add_y_to_x' in train_config['vae'] else False,
        ckpt_path=train_config['vae']['model_path'] if 'model_path' in train_config['vae'] else False,
        upsampler=train_config['vae']['upsampler'] if 'upsampler' in train_config['vae'] else 'nearest',
        sample_mode=True, 
    ).cuda()

    print(model)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

    # bsz = 2
    # x = torch.randn(bsz, in_channels, input_size, input_size).cuda()
    # t = torch.rand(bsz).cuda()
    # y = torch.randn(bsz, model.embed_dim, latent_size, latent_size).cuda()

    # logit = model(x, t, y)
    # print(logit.shape)


    ckpt_dir = train_config['ckpt_path']
    checkpoint = torch.load(ckpt_dir, map_location=lambda storage, loc: storage)
    # if "ema" in checkpoint:  # supports checkpoints from train.py
    #     checkpoint = checkpoint["ema"]
    if "model" in checkpoint:  # supports checkpoints from train.py
        checkpoint = checkpoint["model"]
        checkpoint = {k.replace('module.', ''): v for k, v in checkpoint.items()}
    model.load_state_dict(checkpoint, strict=False)
    model.eval()  # important!
    model.cuda()

    transport = create_transport(
        train_config['flowvae_transport']['path_type'],
        train_config['flowvae_transport']['prediction'],
        train_config['flowvae_transport']['loss_weight'],
        train_config['flowvae_transport']['train_eps'],
        train_config['flowvae_transport']['sample_eps'],
        use_cosine_loss = train_config['flowvae_transport']['use_cosine_loss'] if 'use_cosine_loss' in train_config['flowvae_transport'] else False,
        use_lognorm = train_config['flowvae_transport']['use_lognorm'] if 'use_lognorm' in train_config['flowvae_transport'] else False,
    )  # default: velocity; 

    sampler = Sampler(transport)
    timestep_shift = train_config['flowvae_sample']['timestep_shift'] if 'timestep_shift' in train_config['flowvae_sample'] else 0
    sample_fn = sampler.sample_ode(
        sampling_method=train_config['flowvae_sample']['sampling_method'],
        # num_steps=train_config['flowvae_sample']['num_sampling_steps'] + 1,
        num_steps=1 + 1,
        atol=train_config['flowvae_sample']['atol'],
        rtol=train_config['flowvae_sample']['rtol'],
        reverse=train_config['flowvae_sample']['reverse'],
        timestep_shift=timestep_shift,
    )

    # Setup data
    uncondition = train_config['data']['uncondition'] if 'uncondition' in train_config['data'] else False
    raw_data_dir = train_config['data']['raw_data_dir'] if 'raw_data_dir' in train_config['data'] else None
    crop_size = train_config['data']['image_size']
    raw_img_transform = transforms.Compose([
        transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, crop_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
    ])
    dataset = ImgLatentDataset(
        data_dir=train_config['data']['data_path'],
        latent_norm=train_config['data']['latent_norm'] if 'latent_norm' in train_config['data'] else False,
        latent_multiplier=train_config['data']['latent_multiplier'], 
        raw_data_dir=raw_data_dir, 
        raw_img_transform=raw_img_transform if raw_data_dir is not None else None, 
        raw_img_drop=train_config['data']['raw_img_drop'] if 'raw_img_drop' in train_config['data'] else 0.0,
    )    
    batch_size_per_gpu = 4
    loader = DataLoader(
        dataset,
        batch_size=batch_size_per_gpu,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        drop_last=True
    )

    for latent, label, raw_img in loader:
        x = raw_img.cuda()
        y = latent.cuda()
        model_kwargs = dict(y=y)
        loss_dict = transport.training_losses(
            model, x, model_kwargs, 
            l2_loss=train_config['flowvae_transport']['l2_loss'] if 'l2_loss' in train_config['flowvae_transport'] else True
            )
        xt = loss_dict['xt']
        x1 = loss_dict['x1']
        x0 = loss_dict['x0']
        pred = loss_dict['pred']
        t = loss_dict['t'][:, None, None, None]
        x1_pred = xt + (1-t) * pred
        print(t)
        save_image(xt, "xt.png", nrow=4, normalize=True, value_range=(-1, 1))
        save_image(x1, "x1.png", nrow=4, normalize=True, value_range=(-1, 1))
        save_image(x0, "x0.png", nrow=4, normalize=True, value_range=(-1, 1))
        save_image(x1_pred, "x1_pred.png", nrow=4, normalize=True, value_range=(-1, 1))

        z = x0
        pred_ = model(z, loss_dict['t'], y)
        x1_pred_ = xt + (1-t) * pred
        save_image(x1_pred_, "x1_pred_.png", nrow=4, normalize=True, value_range=(-1, 1))

        # perceptual_loss = LPIPS().cuda().eval()
        # inputs = x1
        # reconstructions = x1_pred
        # p_loss = perceptual_loss(inputs, reconstructions).mean()
        # print(p_loss)


        model_kwargs = dict(y=y)
        model_fn = model.forward
        samples = sample_fn(z, model_fn, **model_kwargs)[-1]
        save_image(samples, "samples.png", nrow=4, normalize=True, value_range=(-1, 1))

        import ipdb;ipdb.set_trace()
        print(1)