import functools
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import math
from einops import rearrange

def hinge_d_loss(logits_real, logits_fake):
    loss_real = torch.mean(F.relu(1.0 - logits_real))
    loss_fake = torch.mean(F.relu(1.0 + logits_fake))
    d_loss = 0.5 * (loss_real + loss_fake)
    return d_loss

def vanilla_d_loss(logits_real, logits_fake):
    d_loss = 0.5 * (
        torch.mean(torch.nn.functional.softplus(-logits_real))
        + torch.mean(torch.nn.functional.softplus(logits_fake))
    )
    return d_loss

def hinge_d_loss_with_exemplar_weights(logits_real, logits_fake, weights):
    assert weights.shape[0] == logits_real.shape[0] == logits_fake.shape[0]
    loss_real = torch.mean(F.relu(1.0 - logits_real), dim=[1, 2, 3])
    loss_fake = torch.mean(F.relu(1.0 + logits_fake), dim=[1, 2, 3])
    loss_real = (weights * loss_real).sum() / weights.sum()
    loss_fake = (weights * loss_fake).sum() / weights.sum()
    d_loss = 0.5 * (loss_real + loss_fake)
    return d_loss

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
    
class Block(nn.Module):
    def __init__(self, c_in, c_out, t_dim=None, k=3, s=1, p=1, bias=False, dropout=0.1):
        super().__init__()
        self.block_in = nn.Sequential(
            nn.Conv2d(c_in, c_out, kernel_size=k, stride=s, padding=p, bias=bias),
            nn.BatchNorm2d(c_out),
            nn.SiLU(True), 
            nn.Dropout(dropout)
        )
        if t_dim is not None:
            self.timestep_proj = nn.Linear(t_dim, c_out)
            self.block_out = nn.Sequential(
                nn.Conv2d(c_out, c_out, kernel_size=1, stride=1, padding=0, bias=bias),
                nn.BatchNorm2d(c_out),
                nn.SiLU(True), 
                nn.Dropout(dropout)
            )
    def forward(self, x, t_in=None):
        x = self.block_in(x)
        if t_in is not None:
            t_in = self.timestep_proj(t_in)
            x = x + t_in[:, :, None, None]
            x = self.block_out(x)
        return x


class Discriminator(nn.Module):
    def __init__(self, disc_model, complex_model=None):
        super().__init__()
        ndf, n_layers = 64, 4
        dropout = disc_model['dropout']
        t_dim = disc_model['timestep_dim'] if complex_model is None else complex_model.hidden_size
        if complex_model is not None:
            ckpt_path = disc_model['complex_model_from']
            checkpoint = torch.load(ckpt_path, map_location=lambda storage, loc: storage)
            checkpoint = {k.replace('module.', ''): v for k, v in checkpoint['model'].items()}
            complex_model.load_state_dict(checkpoint, strict=True)
            complex_layer = disc_model['complex_layer']
            complex_model.blocks = complex_model.blocks[:complex_layer]
            del complex_model.final_layer
        else:
            self.t_embedder = TimestepEmbedder(t_dim)

        input_nc = disc_model['in_c'] if complex_model is None else complex_model.hidden_size
        self.x_embedder = nn.Conv2d(input_nc, ndf, kernel_size=3, stride=2, padding=1) 

        body = []
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            body.append(Block(ndf * nf_mult_prev, ndf * nf_mult, t_dim=t_dim, k=3, s=2, p=1, bias=False, dropout=dropout))

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        body.append(Block(ndf * nf_mult_prev, ndf * nf_mult, t_dim=t_dim, k=3, s=2, p=1, bias=False, dropout=dropout))
        self.body = nn.ModuleList(body)

        # output 1 channel prediction map
        self.out = nn.Sequential(
                nn.AdaptiveAvgPool2d(1), 
                nn.Conv2d(ndf * nf_mult, 1, kernel_size=1, stride=1, padding=0)
            )
        self.initialize_weights()

        # after initializing weights, load from checkpoint
        self.complex_model = complex_model

    def initialize_weights(self):
        def _basic_init(m):
            classname = m.__class__.__name__
            if classname.find('Conv') != -1:
                nn.init.normal_(m.weight.data, 0.0, 0.02)
            elif classname.find('BatchNorm') != -1:
                nn.init.normal_(m.weight.data, 1.0, 0.02)
                nn.init.constant_(m.bias.data, 0)
        self.apply(_basic_init)

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w = self.x_embedder.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.bias, 0)
        # Initialize timestep embedding MLP:
        if hasattr(self, 't_embedder'):
            nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
            nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)
    
    def forward(self, x, t, y):
        if self.complex_model is not None:
            x, t, _ = self.complex_model.forward_feature(x, t, y)
            h = w = int(x.shape[1] ** 0.5)
            x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)
        else:
            t = self.t_embedder(t)                   # (N, D)
        x = self.x_embedder(x)
        for m in self.body:
            x = m(x, t)
        x = self.out(x)
        return x
    
def adopt_weight(weight, global_step, threshold=0, value=0.0):
    if global_step < threshold:
        weight = value
    return weight

class LatentDiscriminator(nn.Module):
    def __init__(
            self, 
            disc_start,
            disc_model, 
            disc_weight=0.5, 
            disc_factor=1.0, 
            disc_loss="hinge",
            complex_model=None, 
            ):
        super().__init__()
        self.discriminator = Discriminator(disc_model, complex_model)
        self.discriminator_iter_start = disc_start
        self.disc_loss = hinge_d_loss if disc_loss == "hinge" else vanilla_d_loss
        self.disc_factor = disc_factor
        self.discriminator_weight = disc_weight

    def calculate_adaptive_weight(self, nll_loss, g_loss, last_layer):
        nll_grads = torch.autograd.grad(nll_loss, last_layer, retain_graph=True)[0]
        g_grads = torch.autograd.grad(g_loss, last_layer, retain_graph=True)[0]

        d_weight = torch.norm(nll_grads) / (torch.norm(g_grads) + 1e-4)
        d_weight = torch.clamp(d_weight, 0.0, 1e4).detach()
        d_weight = d_weight * self.discriminator_weight
        return d_weight
    
    def forward(
        self,
        inputs,
        reconstructions,
        inputs_timestep, 
        recon_timestep,
        labels, 
        main_loss, 
        optimizer_idx,
        global_step,
        last_layer=None,
    ):
        inputs_timestep = inputs_timestep.reshape(-1, )
        recon_timestep = recon_timestep.reshape(-1, )
        # gen
        if optimizer_idx == 0:
            logits_fake = self.discriminator(reconstructions, recon_timestep, labels)
            g_loss = -torch.mean(logits_fake)
            if global_step >= self.discriminator_iter_start:
                if self.disc_factor > 0.0:
                    d_weight = self.calculate_adaptive_weight(
                        main_loss, g_loss, last_layer=last_layer
                    )
                else:
                    d_weight = torch.tensor(1.0)
            else:
                d_weight = torch.tensor(0.0)
                g_loss = torch.tensor(0.0, requires_grad=True)

            disc_factor = adopt_weight(
                self.disc_factor, global_step, threshold=self.discriminator_iter_start
            )
            loss = d_weight * disc_factor * g_loss
            
            log = {
                "g_loss": g_loss.detach().mean().item(),
                "d_weight": d_weight.detach().item(),
                "gen_step_disc_factor": torch.tensor(disc_factor).item(),
            }
            return loss, log
        
        # discriminator training step
        elif optimizer_idx == 1:
            logits_real = self.discriminator(inputs.detach(), inputs_timestep, labels)
            logits_fake = self.discriminator(reconstructions.detach(), recon_timestep, labels)

            disc_factor = adopt_weight(
                self.disc_factor, global_step, threshold=self.discriminator_iter_start
            )

            d_loss = self.disc_loss(logits_real, logits_fake)
            loss = disc_factor * d_loss

            log = {
                "disc_loss": d_loss.clone().detach().mean().item(),
                "disc_step_disc_factor": torch.tensor(disc_factor).item(),
                "logits_real": logits_real.detach().mean().item(),
                "logits_fake": logits_fake.detach().mean().item(),
            }
            return loss, log
    
if __name__ == '__main__':
    import torch
    import yaml
    from copy import deepcopy
    import sys
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    from models import DiT_models

    config_path = 'configs/flow_s_1000kx1024_sdvae_disc_ada_drop0p3_dt0p001.yaml'
    with open(config_path, "r") as f:
        train_config = yaml.safe_load(f)
    # Create model:
    if 'downsample_ratio' in train_config['vae']:
        downsample_ratio = train_config['vae']['downsample_ratio']
    else:
        downsample_ratio = 8
    assert train_config['data']['image_size'] % downsample_ratio == 0, "Image size must be divisible by 8 (for the VAE encoder)."
    latent_size = train_config['data']['image_size'] // downsample_ratio
    in_channels = train_config['model']['in_chans'] if 'in_chans' in train_config['model'] else 4
    model = DiT_models[train_config['model']['model_type']](
        input_size=latent_size,
        num_classes=train_config['data']['num_classes'],
        use_qknorm=train_config['model']['use_qknorm'],
        use_swiglu=train_config['model']['use_swiglu'] if 'use_swiglu' in train_config['model'] else False,
        use_rope=train_config['model']['use_rope'] if 'use_rope' in train_config['model'] else False,
        use_rmsnorm=train_config['model']['use_rmsnorm'] if 'use_rmsnorm' in train_config['model'] else False,
        in_channels=in_channels, 
        use_checkpoint=train_config['model']['use_checkpoint'] if 'use_checkpoint' in train_config['model'] else False,
    )

    disc_model = train_config['discriminator']['disc_model']
    complex_model = deepcopy(model) if train_config['discriminator']['disc_model']['complex'] else None
    discriminator = Discriminator(disc_model, complex_model).cuda()

    bsz = 16
    x = torch.randn(bsz, 4, 32, 32).cuda()
    t = torch.rand(bsz).cuda()
    y = torch.randint(0, 1000, (bsz,)).cuda()
    print(discriminator)
    print(f"Discriminator Parameters: {sum(p.numel() for p in discriminator.parameters()) / 1e6:.2f}M")
    y = discriminator(x, t, y)
    print(y.shape)