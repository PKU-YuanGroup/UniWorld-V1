from typing import List
import torch
import torch.nn as nn
import os
from collections import deque

from ..modules import (
    ResnetBlock2D,
    Conv2d,
    HaarWaveletTransform2D,
    InverseHaarWaveletTransform2D,
    Normalize,
    nonlinearity,
    MlpBlock, 
    TransformerBlock, 
    MMTransformerBlock
)
from einops import rearrange
from ..registry import ModelRegistry
from ..modeling_videobase import VideoBaseAE
from ..utils.module_utils import resolve_str_to_obj
from ..utils.distrib_utils import DiagonalGaussianDistribution
from ..modeling_output import AutoencoderKLOutput, DecoderOutput, ForwardOutput
from diffusers.configuration_utils import register_to_config
import numpy as np
from torch.nn import functional as F

class Encoder(VideoBaseAE):

    @register_to_config
    def __init__(
        self,
        hidden_size: int = 512,
        base_channels: int = 128,
        num_resblocks: int = 2,
        energy_flow_hidden_size: int = 64,
        dropout: float = 0.0,
        attention_type: str = "AttnBlock2D",
        use_attention: bool = True,
        norm_type: str = "groupnorm",
        l1_dowmsample_block: str = "Downsample",
        l1_downsample_wavelet: str = "HaarWaveletTransform2D",
        l2_dowmsample_block: str = "Spatial2xTime2x3DDownsample",
        l2_downsample_wavelet: str = "HaarWaveletTransform3D",
    ) -> None:
        super().__init__()
        self.down1 = nn.Sequential(
            Conv2d(12, base_channels, kernel_size=3, stride=1, padding=1),
            *[
                ResnetBlock2D(
                    in_channels=base_channels,
                    out_channels=base_channels,
                    dropout=dropout,
                    norm_type=norm_type,
                )
                for _ in range(num_resblocks)
            ],
            resolve_str_to_obj(l1_dowmsample_block)(in_channels=base_channels, out_channels=base_channels),
        )
        self.down2 = nn.Sequential(
            Conv2d(
                base_channels + energy_flow_hidden_size,
                base_channels * 2,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            *[
                ResnetBlock2D(
                    in_channels=base_channels * 2,
                    out_channels=base_channels * 2,
                    dropout=dropout,
                    norm_type=norm_type,
                )
                for _ in range(num_resblocks)
            ],
            resolve_str_to_obj(l2_dowmsample_block)(base_channels * 2, base_channels * 2),
        )
        # Connection
        l1_channels = 12
            
        self.connect_l1 = Conv2d(
            l1_channels, energy_flow_hidden_size, kernel_size=3, stride=1, padding=1
        )
        self.connect_l2 = Conv2d(
            12, energy_flow_hidden_size, kernel_size=3, stride=1, padding=1
        )
        # Mid
        mid_layers = [
            ResnetBlock2D(
                in_channels=base_channels * 2 + energy_flow_hidden_size,
                out_channels=base_channels * 4,
                dropout=dropout,
                norm_type=norm_type,
            ),
            ResnetBlock2D(
                in_channels=base_channels * 4,
                out_channels=base_channels * 4,
                dropout=dropout,
                norm_type=norm_type,
            ),
        ]
        if use_attention:
            mid_layers.insert(
                1, resolve_str_to_obj(attention_type)(in_channels=base_channels * 4, norm_type=norm_type)
            )
        self.mid = nn.Sequential(*mid_layers)

        self.norm_out = Normalize(base_channels * 4, norm_type=norm_type)
        self.conv_out = Conv2d(
            base_channels * 4, hidden_size, kernel_size=3, stride=1, padding=1
        )
        
        self.wavelet_transform_in = HaarWaveletTransform2D()
        self.wavelet_transform_l1 = resolve_str_to_obj(l1_downsample_wavelet)()
        self.wavelet_transform_l2 = resolve_str_to_obj(l2_downsample_wavelet)()
        
        
    def forward(self, x):
        coeffs = self.wavelet_transform_in(x)
        
        l1_coeffs = coeffs[:, :3]
        l1_coeffs = self.wavelet_transform_l1(l1_coeffs)
        l1 = self.connect_l1(l1_coeffs)
        l2_coeffs = self.wavelet_transform_l2(l1_coeffs[:, :3])
        l2 = self.connect_l2(l2_coeffs)
        
        h = self.down1(coeffs)
        h = torch.concat([h, l1], dim=1)
        h = self.down2(h)
        h = torch.concat([h, l2], dim=1)
        h = self.mid(h)
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        
        return h, (l1_coeffs, l2_coeffs)

class Decoder(VideoBaseAE):

    @register_to_config
    def __init__(
        self,
        latent_dim: int = 8,
        base_channels: int = 128,
        num_resblocks: int = 2,
        dropout: float = 0.0,
        energy_flow_hidden_size: int = 128,
        attention_type: str = "AttnBlock2D",
        use_attention: bool = True,
        norm_type: str = "groupnorm",
        connect_res_layer_num: int = 1,
        l1_upsample_block: str = "Upsample",
        l1_upsample_wavelet: str = "InverseHaarWaveletTransform2D",
        l2_upsample_block: str = "Spatial2xTime2x3DUpsample",
        l2_upsample_wavelet: str = "InverseHaarWaveletTransform3D",
    ) -> None:
        super().__init__()
        self.energy_flow_hidden_size = energy_flow_hidden_size
    
        self.conv_in = Conv2d(
            latent_dim, base_channels * 4, kernel_size=3, stride=1, padding=1
        )
        mid_layers = [
            ResnetBlock2D(
                in_channels=base_channels * 4,
                out_channels=base_channels * 4,
                dropout=dropout,
                norm_type=norm_type,
            ),
            ResnetBlock2D(
                in_channels=base_channels * 4,
                out_channels=base_channels * 4 + energy_flow_hidden_size,
                dropout=dropout,
                norm_type=norm_type,
            ),
        ]
        if use_attention:
            mid_layers.insert(
                1, resolve_str_to_obj(attention_type)(in_channels=base_channels * 4, norm_type=norm_type)
            )
            
        self.mid = nn.Sequential(*mid_layers)

        self.up2 = nn.Sequential(
            *[
                ResnetBlock2D(
                    in_channels=base_channels * 4,
                    out_channels=base_channels * 4,
                    dropout=dropout,
                    norm_type=norm_type,
                )
                for _ in range(num_resblocks)
            ],
            resolve_str_to_obj(l2_upsample_block)(
                base_channels * 4, base_channels * 4
            ),
            ResnetBlock2D(
                in_channels=base_channels * 4,
                out_channels=base_channels * 4 + energy_flow_hidden_size,
                dropout=dropout,
                norm_type=norm_type,
            ),
        )
        self.up1 = nn.Sequential(
            *[
                ResnetBlock2D(
                    in_channels=base_channels * (4 if i == 0 else 2),
                    out_channels=base_channels * 2,
                    dropout=dropout,
                    norm_type=norm_type,
                )
                for i in range(num_resblocks)
            ],
            resolve_str_to_obj(l1_upsample_block)(in_channels=base_channels * 2, out_channels=base_channels * 2),
            ResnetBlock2D(
                in_channels=base_channels * 2,
                out_channels=base_channels * 2,
                dropout=dropout,
                norm_type=norm_type,
            ),
        )
        self.layer = nn.Sequential(
            *[
                ResnetBlock2D(
                    in_channels=base_channels * (2 if i == 0 else 1),
                    out_channels=base_channels,
                    dropout=dropout,
                    norm_type=norm_type,
                )
                for i in range(2)
            ],
        )
        # Connection
        l1_channels = 12
        self.connect_l1 = nn.Sequential(
            *[
                ResnetBlock2D(
                    in_channels=energy_flow_hidden_size,
                    out_channels=energy_flow_hidden_size,
                    dropout=dropout,
                    norm_type=norm_type,
                )
                for _ in range(connect_res_layer_num)
            ],
            Conv2d(energy_flow_hidden_size, l1_channels, kernel_size=3, stride=1, padding=1),
        )
        self.connect_l2 = nn.Sequential(
            *[
                ResnetBlock2D(
                    in_channels=energy_flow_hidden_size,
                    out_channels=energy_flow_hidden_size,
                    dropout=dropout,
                    norm_type=norm_type,
                )
                for _ in range(connect_res_layer_num)
            ],
            Conv2d(energy_flow_hidden_size, 12, kernel_size=3, stride=1, padding=1),
        )
        # Out
        self.norm_out = Normalize(base_channels, norm_type=norm_type)
        self.conv_out = Conv2d(base_channels, 12, kernel_size=3, stride=1, padding=1)
        
        self.inverse_wavelet_transform_out = InverseHaarWaveletTransform2D()
        self.inverse_wavelet_transform_l1 = resolve_str_to_obj(l1_upsample_wavelet)()
        self.inverse_wavelet_transform_l2 = resolve_str_to_obj(l2_upsample_wavelet)()
        
    def forward(self, z):
        h = self.conv_in(z)
        h = self.mid(h)
        
        l2_coeffs = self.connect_l2(h[:, -self.energy_flow_hidden_size :])
        l2 = self.inverse_wavelet_transform_l2(l2_coeffs)
        
        h = self.up2(h[:, : -self.energy_flow_hidden_size])
        
        l1_coeffs = h[:, -self.energy_flow_hidden_size :]
        l1_coeffs = self.connect_l1(l1_coeffs)
        l1_coeffs[:, :3] = l1_coeffs[:, :3] + l2
        l1 = self.inverse_wavelet_transform_l1(l1_coeffs)

        h = self.up1(h[:, : -self.energy_flow_hidden_size])
        
        h = self.layer(h)
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        h[:, :3] = h[:, :3] + l1
        
        dec = self.inverse_wavelet_transform_out(h)
        return dec, (l1_coeffs, l2_coeffs)




class TextEncoder(VideoBaseAE):

    @register_to_config
    def __init__(
        self,
        vocab_size: int = 1000, 
        hidden_size: int = 512,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        num_textblocks: int = 2, 
    ) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab_size, hidden_size)
        self.block = nn.ModuleList([MlpBlock(hidden_size, mlp_ratio, dropout) for _ in range(num_textblocks)])
        self.out = nn.Sequential(
            nn.LayerNorm(hidden_size), 
            nn.Linear(hidden_size, hidden_size)
            )
        
    def forward(self, input_ids):
        hidden_states = self.embed_tokens(input_ids)
        for m in self.block:
            hidden_states = m(hidden_states)
        hidden_states = self.out(hidden_states)
        return hidden_states

class TextDecoder(VideoBaseAE):

    @register_to_config
    def __init__(
        self,
        latent_dim: int = 8,
        vocab_size: int = 1000, 
        hidden_size: int = 512,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        num_textblocks: int = 2, 
    ) -> None:
        super().__init__()
        self.linear_in = nn.Linear(latent_dim, hidden_size)
        self.block = nn.ModuleList([MlpBlock(hidden_size, mlp_ratio, dropout) for _ in range(num_textblocks)])
        self.out = nn.Sequential(
            nn.LayerNorm(hidden_size), 
            nn.Linear(hidden_size, vocab_size, bias=False)
            )

    def forward(self, z):
        hidden_states = self.linear_in(z)
        for m in self.block:
            hidden_states = m(hidden_states)
        hidden_states = self.out(hidden_states)
        return hidden_states


class MMEncoder(VideoBaseAE):

    @register_to_config
    def __init__(
        self,
        latent_dim: int = 8,
        hidden_size: int = 512,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        num_mmblocks: int = 2, 
    ) -> None:
        super().__init__()
        self.block = nn.ModuleList([MMTransformerBlock(hidden_size, num_heads, mlp_ratio, dropout) for _ in range(num_mmblocks)])
        self.out = nn.Sequential(
            nn.LayerNorm(hidden_size), 
            nn.Linear(hidden_size, latent_dim * 2)
            )
        
    def forward(self, x, x_t):
        for m in self.block:
            x, x_t = m(x, x_t)
        if x is not None:
            B, C, H, W = x.shape
            x = rearrange(x, 'b c h w -> b (h w) c')
            x = self.out(x)
            x = rearrange(x, 'b (h w) c -> b c h w', h=H, w=W)
        if x_t is not None:
            x_t = self.out(x_t)
        return x, x_t

class ClipHead(nn.Module):
    def __init__(self, hidden_size, projection_dim):
        super().__init__()
        self.visual_projection = nn.Linear(hidden_size, projection_dim, bias=False) 
        self.text_projection = nn.Linear(hidden_size, projection_dim, bias=False)
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07)) 

    def forward(self, h=None, h_t=None):
        image_features, text_features = None, None
        if h is not None:
            image_features = F.adaptive_avg_pool2d(h, 1).squeeze(2).squeeze(2)
            image_features = self.visual_projection(image_features)
            image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
        if h_t is not None:
            text_features = h_t[:, 0]
            text_features = self.text_projection(text_features)
            text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)
        clip_feature = (image_features, text_features, self.logit_scale.exp())
        return clip_feature
    
@ModelRegistry.register("MMVAE")
class MMVAEModel(VideoBaseAE):

    @register_to_config
    def __init__(
        self,
        latent_dim: int = 8,
        base_channels: int = 128,
        encoder_num_resblocks: int = 2,
        encoder_energy_flow_hidden_size: int = 64,
        decoder_num_resblocks: int = 2,
        decoder_energy_flow_hidden_size: int = 128,
        attention_type: str = "AttnBlock2D",
        use_attention: bool = True,
        dropout: float = 0.0,
        norm_type: str = "groupnorm",
        connect_res_layer_num: int = 1,
        scale: List[float] = [0.18215, 0.18215, 0.18215, 0.18215],
        shift: List[float] = [0, 0, 0, 0],
        # Module config
        l1_dowmsample_block: str = "Downsample",
        l1_downsample_wavelet: str = "HaarWaveletTransform2D",
        l2_dowmsample_block: str = "Spatial2xTime2x3DDownsample",
        l2_downsample_wavelet: str = "HaarWaveletTransform3D",
        l1_upsample_block: str = "Upsample",
        l1_upsample_wavelet: str = "InverseHaarWaveletTransform2D",
        l2_upsample_block: str = "Spatial2xTime2x3DUpsample",
        l2_upsample_wavelet: str = "InverseHaarWaveletTransform3D",
        # text config
        vocab_size: int = 1000, 
        hidden_size: int = 512,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        num_textblocks: int = 2, 
        # mm config
        num_mmblocks: int = 4, 
        use_clip_loss: bool = False, 
        projection_dim: int = 768, 
        # 
        train_image: bool = True, 
        train_text: bool = True, 
    ) -> None:
        super().__init__()

        self.train_image = train_image
        self.train_text = train_text

        if self.train_image:
            self.encoder = Encoder(
                hidden_size=hidden_size,
                base_channels=base_channels,
                num_resblocks=encoder_num_resblocks,
                energy_flow_hidden_size=encoder_energy_flow_hidden_size,
                dropout=dropout,
                use_attention=use_attention,
                norm_type=norm_type,
                l1_dowmsample_block=l1_dowmsample_block,
                l1_downsample_wavelet=l1_downsample_wavelet,
                l2_dowmsample_block=l2_dowmsample_block,
                l2_downsample_wavelet=l2_downsample_wavelet,
                attention_type=attention_type
            )
            self.decoder = Decoder(
                latent_dim=latent_dim,
                base_channels=base_channels,
                num_resblocks=decoder_num_resblocks,
                energy_flow_hidden_size=decoder_energy_flow_hidden_size,
                dropout=dropout,
                use_attention=use_attention,
                norm_type=norm_type,
                connect_res_layer_num=connect_res_layer_num,
                l1_upsample_block=l1_upsample_block,
                l1_upsample_wavelet=l1_upsample_wavelet,
                l2_upsample_block=l2_upsample_block,
                l2_upsample_wavelet=l2_upsample_wavelet,
                attention_type=attention_type
            )

        if self.train_text:
            self.text_encoder = TextEncoder(
                vocab_size=vocab_size, 
                hidden_size=hidden_size, 
                num_heads=num_heads, 
                mlp_ratio=mlp_ratio, 
                dropout=dropout, 
                num_textblocks=num_textblocks
                )

            self.text_decoder = TextDecoder(
                latent_dim=latent_dim, 
                vocab_size=vocab_size, 
                hidden_size=hidden_size, 
                num_heads=num_heads, 
                mlp_ratio=mlp_ratio, 
                dropout=dropout, 
                num_textblocks=num_textblocks
                )
        
        self.mm_encoder = MMEncoder(
            latent_dim=latent_dim, 
            hidden_size=hidden_size, 
            num_heads=num_heads, 
            mlp_ratio=mlp_ratio, 
            dropout=dropout, 
            num_mmblocks=num_mmblocks
            )

        self.use_clip_loss = use_clip_loss
        if self.use_clip_loss:
            self.clip_head = ClipHead(hidden_size, projection_dim)
        
    def get_encoder(self):
        if self.train_image and self.train_text:
            return [self.encoder, self.text_encoder, self.mm_encoder]
        if self.train_image:
            return [self.encoder, self.mm_encoder]
        if self.train_text:
            return [self.text_encoder, self.mm_encoder]

    def get_clip_param(self):
        assert self.use_clip_loss
        return [self.clip_head]

    def get_decoder(self):
        if self.train_image and self.train_text:
            return [self.decoder, self.text_decoder]
        if self.train_image:
            return [self.decoder]
        if self.train_text:
            return [self.text_decoder]

    def get_encoder_text_features(self, input_ids, pre_fea=False):
        h_t = input_ids if pre_fea else self.text_encoder(input_ids)
        encoder_text_features = h_t[:, 0]
        encoder_text_features = encoder_text_features / encoder_text_features.norm(p=2, dim=-1, keepdim=True)
        return encoder_text_features

    def get_encoder_img_features(self, x, pre_fea=False):
        h = x if pre_fea else self.encoder(x)[0]
        encoder_image_features = F.adaptive_avg_pool2d(h, 1).squeeze(2).squeeze(2)
        encoder_image_features = encoder_image_features / encoder_image_features.norm(p=2, dim=-1, keepdim=True)
        return encoder_image_features

    def get_clip_text_features(self, input_ids):
        h_t = self.text_encoder(input_ids)
        text_features = self.clip_head(h_t=h_t)[1]
        return text_features

    def get_clip_img_features(self, x):
        h, (l1, l2) = self.encoder(x)
        img_features = self.clip_head(h=h)[0]
        return img_features

    def encode(self, x=None, input_ids=None):
        assert x is not None or input_ids is not None
        h, h_t, l1, l2 = None, None, None, None
        enc_img_features, enc_text_features = None, None
        if x is not None and self.train_image:
            h, (l1, l2) = self.encoder(x)
            enc_img_features = self.get_encoder_img_features(h, pre_fea=True)
        if input_ids is not None and self.train_text:
            h_t = self.text_encoder(input_ids)
            enc_text_features = self.get_encoder_text_features(h_t, pre_fea=True)
        enc_features = (enc_img_features, enc_text_features)


        clip_feature = None
        if self.use_clip_loss and self.train_image and self.train_text:
            clip_feature = self.clip_head(h=h, h_t=h_t)

        h, h_t = self.mm_encoder(h, h_t)
        posterior = DiagonalGaussianDistribution(h, h_t)
        return AutoencoderKLOutput(latent_dist=posterior, extra_output=(l1, l2), enc_features=enc_features, clip_feature=clip_feature)


    def decode(self, z, z_t):
        assert z is not None or z_t is not None
        dec, dec_t, l1, l2 = None, None, None, None
        if z is not None and self.train_image:
            dec, (l1, l2) = self.decoder(z)
        if z_t is not None and self.train_text:
            dec_t = self.text_decoder(z_t)
        return DecoderOutput(sample=(dec, dec_t), extra_output=(l1, l2))
    

    def forward(self, input=None, input_ids=None, sample_posterior=True):
        encode_output = self.encode(input, input_ids)
        posterior, (enc_l1, enc_l2), enc_features, clip_feature = encode_output.latent_dist, encode_output.extra_output, encode_output.enc_features, encode_output.clip_feature
        
        if sample_posterior:
            z, z_t = posterior.sample()
            
        decode_output = self.decode(z, z_t)
        (dec, dec_t), (dec_l1, dec_l2) = decode_output.sample, decode_output.extra_output
        return ForwardOutput(sample=(dec, dec_t), latent_dist=posterior, extra_output=(enc_l1, dec_l1, enc_l2, dec_l2), enc_features=enc_features, clip_feature=clip_feature)

    def get_last_layer(self):
        if hasattr(self.decoder.conv_out, "conv"):
            return self.decoder.conv_out.conv.weight
        else:
            return self.decoder.conv_out.weight

    def init_from_ckpt(self, path, ignore_keys=list()):
        sd = torch.load(path, map_location="cpu")
        print("init from " + path)

        if (
            "ema_state_dict" in sd
            and len(sd["ema_state_dict"]) > 0
            and os.environ.get("NOT_USE_EMA_MODEL", 0) == 0
        ):
            print("Load from ema model!")
            sd = sd["ema_state_dict"]
            sd = {key.replace("module.", ""): value for key, value in sd.items()}
        elif "state_dict" in sd:
            print("Load from normal model!")
            if "gen_model" in sd["state_dict"]:
                sd = sd["state_dict"]["gen_model"]
            else:
                sd = sd["state_dict"]

        keys = list(sd.keys())

        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]

        missing_keys, unexpected_keys = self.load_state_dict(sd, strict=False)