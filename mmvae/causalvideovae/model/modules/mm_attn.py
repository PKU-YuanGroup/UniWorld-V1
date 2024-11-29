import torch
import torch.nn as nn
from timm.models.vision_transformer import PatchEmbed, Mlp
from torch.nn import functional as F
from einops import rearrange

class Attention(nn.Module):
    def __init__(
            self,
            dim: int,
            num_heads: int = 8,
            qkv_bias: bool = False,
            qk_norm: bool = False,
            attn_drop: float = 0.,
            proj_drop: float = 0.,
            norm_layer: nn.Module = nn.LayerNorm,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()
        q, k = self.q_norm(q), self.k_norm(k)

        x = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.attn_drop.p if self.training else 0.,
        )

        x = x.transpose(1, 2).reshape(B, N, C).contiguous()
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
    
class MlpBlock(nn.Module):
    def __init__(self, hidden_size, mlp_ratio=4.0, dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=dropout)

    def forward(self, features):
        
        assert features.dim() == 3

        ffn_out = self.mlp(self.norm1(features))
        features = features + ffn_out

        return features

class TransformerBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, dropout=0.0):
        super().__init__()
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True)
        self.norm1 = nn.LayerNorm(hidden_size)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=dropout)
        self.norm2 = nn.LayerNorm(hidden_size)

    def forward(self, features):
        
        assert features.dim() == 3

        attn_out = self.attn(self.norm1(features))
        features = features + attn_out

        ffn_out = self.mlp(self.norm2(features))
        features = features + ffn_out

        return features
    
class MMTransformerBlock(TransformerBlock):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, img=None, text=None):
        assert img is not None or text is not None
        if img is not None:
            B, C, H, W = img.shape
            img = rearrange(img, 'b c h w -> b (h w) c')
        
        if text is not None:
            assert text.ndim == 3
        
        if img is not None and text is None:
            img_features = super().forward(img)
            img_features = rearrange(img_features, 'b (h w) c -> b c h w', h=H, w=W)
            return img_features, None
        
        if img is None and text is not None:
            text_features = super().forward(text)
            return None, text_features
        
        img_tokens = img.shape[1]
        features = torch.cat([img, text], dim=1) 
        features = super().forward(features)
        img_features, text_features = torch.split(features, img_tokens, dim=1)
        img_features = rearrange(img_features, 'b (h w) c -> b c h w', h=H, w=W)
        return img_features, text_features
