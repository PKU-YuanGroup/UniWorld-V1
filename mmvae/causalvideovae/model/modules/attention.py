import torch.nn as nn
from .normalize import Normalize
from .conv import CausalConv3d
import torch
import torch.nn.functional as F
from einops import rearrange

class AttnBlock2D(nn.Module):
    def __init__(self, in_channels, norm_type="groupnorm"):
        super().__init__()
        self.in_channels = in_channels

        self.norm = Normalize(in_channels, norm_type=norm_type)
        self.q = torch.nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
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

        # compute attention
        b, c, h, w = q.shape


        q = rearrange(q.contiguous(), 'b c h w -> b 1 (h w) c')
        k = rearrange(k.contiguous(), 'b c h w -> b 1 (h w) c')
        v = rearrange(v.contiguous(), 'b c h w -> b 1 (h w) c')

        attn_output = F.scaled_dot_product_attention(
            q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False
        )
        attn_output = rearrange(attn_output, 'b 1 (h w) c -> b c h w', h=h, w=w)


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

        h_ = self.proj_out(attn_output)

        return x + h_

class AttnBlock3D(nn.Module):
    """Compatible with old versions, there are issues, use with caution."""
    def __init__(self, in_channels, norm_type="groupnorm"):
        super().__init__()
        self.in_channels = in_channels

        self.norm = Normalize(in_channels, norm_type=norm_type)
        self.q = CausalConv3d(in_channels, in_channels, kernel_size=1, stride=1)
        self.k = CausalConv3d(in_channels, in_channels, kernel_size=1, stride=1)
        self.v = CausalConv3d(in_channels, in_channels, kernel_size=1, stride=1)
        self.proj_out = CausalConv3d(in_channels, in_channels, kernel_size=1, stride=1)

    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # compute attention
        b, c, t, h, w = q.shape
        q = q.reshape(b * t, c, h * w)
        q = q.permute(0, 2, 1)  # b,hw,c
        k = k.reshape(b * t, c, h * w)  # b,c,hw
        w_ = torch.bmm(q, k)  # b,hw,hw    w[b,i,j]=sum_c q[b,i,c]k[b,c,j]
        w_ = w_ * (int(c) ** (-0.5))
        w_ = torch.nn.functional.softmax(w_, dim=2)

        # attend to values
        v = v.reshape(b * t, c, h * w)
        w_ = w_.permute(0, 2, 1)  # b,hw,hw (first hw of k, second of q)
        h_ = torch.bmm(v, w_)  # b, c,hw (hw of q) h_[b,c,j] = sum_i v[b,c,i] w_[b,i,j]
        h_ = h_.reshape(b, c, t, h, w)

        h_ = self.proj_out(h_)

        return x + h_


class AttnBlock3DFix(nn.Module):

    def __init__(self, in_channels, norm_type="groupnorm"):
        super().__init__()
        self.in_channels = in_channels

        self.norm = Normalize(in_channels, norm_type=norm_type)
        self.q = CausalConv3d(in_channels, in_channels, kernel_size=1, stride=1)
        self.k = CausalConv3d(in_channels, in_channels, kernel_size=1, stride=1)
        self.v = CausalConv3d(in_channels, in_channels, kernel_size=1, stride=1)
        self.proj_out = CausalConv3d(in_channels, in_channels, kernel_size=1, stride=1)

    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        b, c, t, h, w = q.shape
        q = q.permute(0, 2, 3, 4, 1).reshape(b * t, h * w, c).contiguous()
        k = k.permute(0, 2, 3, 4, 1).reshape(b * t, h * w, c).contiguous()
        v = v.permute(0, 2, 3, 4, 1).reshape(b * t, h * w, c).contiguous()
        
        if x.device.type == "cpu":
            attn = torch.bmm(q, k.transpose(-2, -1)) * (c ** -0.5)
            attn = F.softmax(attn, dim=-1)

            attn_output = torch.bmm(attn, v)
            attn_output = attn_output.reshape(b, t, h, w, c).permute(0, 4, 1, 2, 3)
        else:        
            # attn_output = xops.memory_efficient_attention(
            #     q, k, v,
            #     scale=c ** -0.5
            # )
            q = q.view(b * t, -1, 1, c).transpose(1, 2)
            k = k.view(b * t, -1, 1, c).transpose(1, 2)
            v = v.view(b * t, -1, 1, c).transpose(1, 2)

            attn_output = F.scaled_dot_product_attention(
                q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False
            )
            attn_output = attn_output.transpose(1, 2).reshape(b * t, -1, 1 * c)

            
        attn_output = attn_output.reshape(b, t, h, w, c).permute(0, 4, 1, 2, 3)
        h_ = self.proj_out(attn_output)

        return x + h_

class AttnBlock2Plus1D(nn.Module):
    def __init__(self, in_channels, norm_type="groupnorm"):
        super().__init__()
        self.in_channels = in_channels

        self.norm = Normalize(in_channels, norm_type=norm_type)
        self.q = CausalConv3d(in_channels, in_channels, kernel_size=1, stride=1)
        self.k = CausalConv3d(in_channels, in_channels, kernel_size=1, stride=1)
        self.v = CausalConv3d(in_channels, in_channels, kernel_size=1, stride=1)
        self.proj_out = CausalConv3d(in_channels, in_channels, kernel_size=1, stride=1)

    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        b, c, t, h, w = q.shape

        # Compute attention over spatial dimensions (h, w) for each timestep
        q_spatial = rearrange(q, 'b c t h w -> (b t) (h w) c').contiguous()
        k_spatial = rearrange(k, 'b c t h w -> (b t) (h w) c').contiguous()
        v_spatial = rearrange(v, 'b c t h w -> (b t) (h w) c').contiguous()

        if x.device.type == "cpu":
            attn_spatial = torch.bmm(q_spatial, k_spatial.transpose(-2, -1)) * (c ** -0.5)
            attn_spatial = F.softmax(attn_spatial, dim=-1)
            attn_output_spatial = torch.bmm(attn_spatial, v_spatial)
        else:
            attn_output_spatial = xops.memory_efficient_attention(
                q_spatial, k_spatial, v_spatial,
                scale=c ** -0.5
            )

        attn_output_spatial = rearrange(attn_output_spatial, '(b t) (h w) c -> b c t h w', b=b, t=t, h=h, w=w)

        # Compute attention over temporal dimension (t) for each spatial position
        q_temporal = rearrange(q, 'b c t h w -> (b h w) t c').contiguous()
        k_temporal = rearrange(k, 'b c t h w -> (b h w) t c').contiguous()
        v_temporal = rearrange(v, 'b c t h w -> (b h w) t c').contiguous()

        if x.device.type == "cpu":
            attn_temporal = torch.bmm(q_temporal, k_temporal.transpose(-2, -1)) * (c ** -0.5)
            attn_temporal = F.softmax(attn_temporal, dim=-1)
            attn_output_temporal = torch.bmm(attn_temporal, v_temporal)
        else:
            attn_output_temporal = xops.memory_efficient_attention(
                q_temporal, k_temporal, v_temporal,
                scale=c ** -0.5
            )

        attn_output_temporal = rearrange(attn_output_temporal, '(b h w) t c -> b c t h w', b=b, h=h, w=w)

        # Sum the spatial and temporal attention results
        attn_output = attn_output_spatial + attn_output_temporal

        h_ = self.proj_out(attn_output)

        return x + h_
