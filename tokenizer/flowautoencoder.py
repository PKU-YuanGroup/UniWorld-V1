import os
import torch
import math
from tqdm import tqdm
import numpy as np
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from timm.models.vision_transformer import PatchEmbed, Mlp
from einops import rearrange
import torch.nn.functional as F


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
    # swish
    return x * torch.sigmoid(x)


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
        return LayerNorm(num_channels=in_channels, eps=1e-6)

class Upsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            self.conv = torch.nn.Conv2d(
                in_channels, in_channels, kernel_size=3, stride=1, padding=1
            )

    def forward(self, x):
        x = torch.nn.functional.interpolate(x, scale_factor=2.0, mode="nearest")
        if self.with_conv:
            x = self.conv(x)
        return x


class Downsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            # no asymmetric padding in torch conv, must do it ourselves
            self.conv = torch.nn.Conv2d(
                in_channels, in_channels, kernel_size=3, stride=2, padding=0
            )

    def forward(self, x):
        if self.with_conv:
            pad = (0, 1, 0, 1)
            x = torch.nn.functional.pad(x, pad, mode="constant", value=0)
            x = self.conv(x)
        else:
            x = torch.nn.functional.avg_pool2d(x, kernel_size=2, stride=2)
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


class Encoder(nn.Module):
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
        double_z=True,
        **ignore_kwargs,
    ):
        super().__init__()
        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels

        # downsampling
        self.conv_in = torch.nn.Conv2d(
            in_channels, self.ch, kernel_size=3, stride=1, padding=1
        )

        curr_res = resolution
        in_ch_mult = (1,) + tuple(ch_mult)
        self.down = nn.ModuleList()
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = ch * in_ch_mult[i_level]
            block_out = ch * ch_mult[i_level]
            for i_block in range(self.num_res_blocks):
                block.append(
                    ResnetBlock(
                        in_channels=block_in,
                        out_channels=block_out,
                        temb_channels=self.temb_ch,
                        dropout=dropout,
                    )
                )
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(AttnBlock(block_in))
            down = nn.Module()
            down.block = block
            down.attn = attn
            if i_level != self.num_resolutions - 1:
                down.downsample = Downsample(block_in, resamp_with_conv)
                curr_res = curr_res // 2
            self.down.append(down)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(
            in_channels=block_in,
            out_channels=block_in,
            temb_channels=self.temb_ch,
            dropout=dropout,
        )
        self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_2 = ResnetBlock(
            in_channels=block_in,
            out_channels=block_in,
            temb_channels=self.temb_ch,
            dropout=dropout,
        )

        # end
        self.norm_out = Normalize(block_in)
        self.conv_out = torch.nn.Conv2d(
            block_in,
            2 * z_channels if double_z else z_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )

    def forward(self, x):
        # assert x.shape[2] == x.shape[3] == self.resolution, "{}, {}, {}".format(x.shape[2], x.shape[3], self.resolution)

        # timestep embedding
        temb = None

        # downsampling
        hs = [self.conv_in(x)]
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](hs[-1], temb)
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
                hs.append(h)
            if i_level != self.num_resolutions - 1:
                hs.append(self.down[i_level].downsample(hs[-1]))

        # middle
        h = hs[-1]
        h = self.mid.block_1(h, temb)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)

        # end
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        return h


class Decoder(nn.Module):
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
        )
        self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_2 = ResnetBlock(
            in_channels=block_in,
            out_channels=block_in,
            temb_channels=self.temb_ch,
            dropout=dropout,
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
                    )
                )
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(AttnBlock(block_in))
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0:
                up.upsample = Upsample(block_in, resamp_with_conv)
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
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)
            if i_level != 0:
                h = self.up[i_level].upsample(h)
                temb_idx += 1

        # end
        if self.give_pre_end:
            return h

        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        return h


class DiagonalGaussianDistribution(object):
    def __init__(self, parameters, deterministic=False):
        self.parameters = parameters
        self.mean, self.logvar = torch.chunk(parameters, 2, dim=1)
        self.logvar = torch.clamp(self.logvar, -30.0, 20.0)
        self.deterministic = deterministic
        self.std = torch.exp(0.5 * self.logvar)
        self.var = torch.exp(self.logvar)
        if self.deterministic:
            self.var = self.std = torch.zeros_like(self.mean).to(
                device=self.parameters.device
            )

    def sample(self):
        x = self.mean + self.std * torch.randn(self.mean.shape).to(
            device=self.parameters.device
        )
        return x

    def kl(self, other=None):
        if self.deterministic:
            return torch.Tensor([0.0])
        else:
            if other is None:
                return 0.5 * torch.sum(
                    torch.pow(self.mean, 2) + self.var - 1.0 - self.logvar,
                    dim=[1, 2, 3],
                )
            else:
                return 0.5 * torch.sum(
                    torch.pow(self.mean - other.mean, 2) / other.var
                    + self.var / other.var
                    - 1.0
                    - self.logvar
                    + other.logvar,
                    dim=[1, 2, 3],
                )

    def nll(self, sample, dims=[1, 2, 3]):
        if self.deterministic:
            return torch.Tensor([0.0])
        logtwopi = np.log(2.0 * np.pi)
        return 0.5 * torch.sum(
            logtwopi + self.logvar + torch.pow(sample - self.mean, 2) / self.var,
            dim=dims,
        )

    def mode(self):
        return self.mean

def load_weights_with_shape_check(model, checkpoint):
    model_state_dict = model.state_dict()
    # check shape and load weights
    for name, param in checkpoint.items():
        if name in model_state_dict:
            if param.shape == model_state_dict[name].shape:
                model_state_dict[name].copy_(param)
            else:
                print(f"Skipping loading parameter '{name}' due to shape mismatch: "
                    f"checkpoint shape {param.shape}, model shape {model_state_dict[name].shape}")
        else:
            print(f"Parameter '{name}' not found in model, skipping.")
    # load state dict
    model.load_state_dict(model_state_dict, strict=False)
    
    return model

class FlowAutoencoderKL(nn.Module):
    def __init__(
            self, 
            embed_dim, 
            ch_mult, 
            input_size=256,
            patch_size=8,
            ch=128, 
            temb_ch=512, 
            use_variational=True, 
            ckpt_path=None, 
            model_type='vavae', 
            train_decoder=True, 
            norm_type='groupnorm', 
            ):
        super().__init__()
        self.embed_dim = embed_dim
        self.model_type = model_type
        self.use_variational = use_variational
        mult = 2 if self.use_variational else 1

        # encode
        self.encoder = Encoder(ch_mult=ch_mult, z_channels=embed_dim)
        self.quant_conv = torch.nn.Conv2d(2 * embed_dim, mult * embed_dim, 1)

        # decode
        if model_type == 'vavae' or model_type == 'sdvae':
            self.decoder = Decoder(
                ch=ch, ch_mult=ch_mult, z_channels=ch * ch_mult[-1], 
                attn_resolutions=(16,), temb_ch=temb_ch, norm_type=norm_type
                )
        elif model_type == 'marvae':
            self.decoder = Decoder(
                ch=ch, ch_mult=ch_mult, z_channels=ch * ch_mult[-1], 
                attn_resolutions=(), temb_ch=temb_ch, norm_type=norm_type
                )

        # flow
        hidden_size = ch * ch_mult[-1]
        self.t_embedder = TimestepEmbedder(temb_ch)
        self.y_embedder = nn.ModuleList([
            torch.nn.Conv2d(embed_dim, temb_ch, 1) if i == 0 else Upsample(temb_ch, with_conv=True) 
            for i in range(len(ch_mult))
            ])
        self.x_embedder = PatchEmbed(input_size, patch_size, 3, hidden_size, bias=True)
        num_patches = self.x_embedder.num_patches
        # Will use fixed sin-cos embedding:
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, hidden_size), requires_grad=False)
        self.dit_initialize_weights()

        # init 
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path)
        
        # freeze encode
        self.train_decoder = train_decoder
        self.encoder.requires_grad_(False)
        self.quant_conv.requires_grad_(False)

    def init_from_ckpt(self, path):
        if self.model_type == 'vavae':
            sd = torch.load(path, map_location="cpu")
            if 'state_dict' in sd.keys():
                sd = sd['state_dict']
            sd = {k: v for k, v in sd.items() if 'foundation_model.model' not in k and 'loss' not in k}
            load_weights_with_shape_check(self, sd)
        elif self.model_type == 'sdvae':
            from safetensors.torch import load_file
            sd = load_file(path)
            load_weights_with_shape_check(self, sd)
        elif self.model_type == 'marvae':
            sd = torch.load(path, map_location="cpu")["model"]
            load_weights_with_shape_check(self, sd)
        else:
            raise ValueError(f"Invalid model type: {self.model_type}")
    
    def dit_initialize_weights(self):

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

    def decode(self, z, t, y):
        z = self.x_embedder(z) + self.pos_embed
        h = w = int(z.shape[1] ** 0.5)
        z = rearrange(z, 'b (h w) c -> b c h w', h=h, w=w).contiguous()  # B C H W
        t = self.t_embedder(t)[:, :, None, None]    # (B, C) -> # (B, C, 1, 1)
        c = []
        for m in self.y_embedder:
            y = m(y)    # B C H W -> B C 2*H 2*W -> B C 4*H 4*W
            c.append(t + y)

        dec = self.decoder(z, c) # B 3 256 256
        return dec

    @torch.compile
    def forward(self, x, t=None, y=None, **kwargs):
        x = self.decode(x, t, y)
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
    return FlowAutoencoderKL(embed_dim=32, patch_size=16, ch_mult=(1, 1, 2, 2, 4), model_type='vavae', **kwargs)

def FlowSDVAE(**kwargs):
    return FlowAutoencoderKL(embed_dim=4, patch_size=8, ch_mult=(1, 2, 4, 4), model_type='sdvae', **kwargs)

def FlowMARVAE(**kwargs):
    return FlowAutoencoderKL(embed_dim=16, patch_size=16, ch_mult=(1, 1, 2, 2, 4), model_type='marvae', **kwargs)

FlowVAE_models = {
    'FlowVAVAE':  FlowVAVAE,  'FlowSDVAE':  FlowSDVAE,  'FlowMARVAE':  FlowMARVAE,  
}


if __name__ == '__main__':
    import yaml

    config_path = 'configs/flowvavae_s_100kx1024.yaml'
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
        train_decoder=train_config['vae']['train_decoder'] if 'train_decoder' in train_config['vae'] else True,
        norm_type=train_config['vae']['norm_type'] if 'norm_type' in train_config['vae'] else 'groupnorm',
        ckpt_path=train_config['vae']['model_path'] if 'model_path' in train_config['vae'] else False,
    ).cuda()

    print(model)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

    bsz = 2
    x = torch.randn(bsz, in_channels, input_size, input_size).cuda()
    t = torch.rand(bsz).cuda()
    y = torch.randn(bsz, model.embed_dim, latent_size, latent_size).cuda()

    logit = model(x, t, y)
    print(logit.shape)