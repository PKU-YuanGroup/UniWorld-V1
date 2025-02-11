from timm.models.vision_transformer import Block
import torch
from torch import nn
import torch.nn.functional as F
    
class QFormer(nn.Module):
    def __init__(self, num_layers, num_queries, embed_dim, num_heads, mlp_ratio, qkv_bias=True, use_qknorm=True):
        super().__init__()
        self.num_queries = num_queries
        self.query_embed = nn.Parameter(torch.randn(1, num_queries, embed_dim))
        self.layers = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias, qk_norm=use_qknorm)
            for _ in range(num_layers)
        ])
    @torch.compile
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        queries = self.query_embed.expand(batch_size, -1, -1)  # (batch_size, num_queries, embed_dim)
        x = torch.cat([queries, x], dim=1)
        for layer in self.layers:
            x = layer(x)
        return x[:, :self.num_queries, :].mean(1)