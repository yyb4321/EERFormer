import torch
import torch.nn as nn
from einops import rearrange
from einops.layers.torch import Rearrange
from torch.nn import functional as F

# from .networks.segformer import *
# from .CFGCN import CFGCN
from .segformer import *
from .masag import MultiScaleGatedAttn
from .merit_lib.networks import MaxViT4Out_Small
# from segformer import *
# from attentions import MultiScaleGatedAttn
from .modules import *
from timm.models.layers import DropPath, to_2tuple
import math
from torchvision.transforms.functional import rgb_to_grayscale
from torchvision.transforms import ToPILImage

# from .MIST import *
##################################
#
#            Modules
#
##################################


class DWConvLKA(nn.Module):
    def __init__(self, dim=768):
        super(DWConvLKA, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x):
        x = self.dwconv(x)
        return x


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0., linear=False):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
        self.dwconv = DWConvLKA(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)
        self.linear = linear
        if self.linear:
            self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.fc1(x)
        if self.linear:
            x = self.relu(x)
        x = self.dwconv(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class AttentionModule(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv0 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
        self.conv_spatial = nn.Conv2d(
            dim, dim, 7, stride=1, padding=9, groups=dim, dilation=3)
        self.conv1 = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        u = x.clone()
        attn = self.conv0(x)
        attn = self.conv_spatial(attn)
        attn = self.conv1(attn)
        return u * attn

class FocalModulation(nn.Module):
    def __init__(self, dim, focal_window, focal_level, focal_factor=2, bias=True, proj_drop=0.,
                 use_postln_in_modulation=False, normalize_modulator=False):
        super().__init__()

        self.dim = dim
        self.focal_window = focal_window
        self.focal_level = focal_level
        self.focal_factor = focal_factor
        self.use_postln_in_modulation = use_postln_in_modulation
        self.normalize_modulator = normalize_modulator

        self.f = nn.Linear(dim, 2 * dim + (self.focal_level + 1), bias=bias)
        self.h = nn.Conv2d(dim, dim, kernel_size=1, stride=1, bias=bias)

        self.act = nn.GELU()
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.focal_layers = nn.ModuleList()

        self.kernel_sizes = []
        for k in range(self.focal_level):
            kernel_size = self.focal_factor * k + self.focal_window
            self.focal_layers.append(
                nn.Sequential(
                    nn.Conv2d(dim, dim, kernel_size=kernel_size, stride=1,
                              groups=dim, padding=kernel_size // 2, bias=False),
                    nn.GELU(),
                )
            )
            self.kernel_sizes.append(kernel_size)
        if self.use_postln_in_modulation:
            self.ln = nn.LayerNorm(dim)

    def forward(self, x):
        """
        Args:
            x: input features with shape of (B, H, W, C)
        """
        C = x.shape[-1]

        # pre linear projection
        x = self.f(x)
        x= x.permute(0, 3, 1, 2).contiguous()
        q, ctx, self.gates = torch.split(x, (C, C, self.focal_level + 1), 1)

        # context aggreation
        ctx_all = 0
        for l in range(self.focal_level):
            ctx = self.focal_layers[l](ctx)
            ctx_all = ctx_all + ctx * self.gates[:, l:l + 1]
        ctx_global = self.act(ctx.mean(2, keepdim=True).mean(3, keepdim=True))
        ctx_all = ctx_all + ctx_global * self.gates[:, self.focal_level:]

        # normalize context
        if self.normalize_modulator:
            ctx_all = ctx_all / (self.focal_level + 1)

        # focal modulation
        self.modulator = self.h(ctx_all)
        x_out = q * self.modulator
        x_out = x_out.permute(0, 2, 3, 1).contiguous()
        if self.use_postln_in_modulation:
            x_out = self.ln(x_out)

        # post linear porjection
        x_out = self.proj(x_out)
        x_out = self.proj_drop(x_out)
        return x_out

    def extra_repr(self) -> str:
        return f'dim={self.dim}'

    def flops(self, N):
        # calculate flops for 1 window with token length of N
        flops = 0

        flops += N * self.dim * (self.dim * 2 + (self.focal_level + 1))

        # focal convolution
        for k in range(self.focal_level):
            flops += N * (self.kernel_sizes[k] ** 2 + 1) * self.dim

        # global gating
        flops += N * 1 * self.dim

        #  self.linear
        flops += N * self.dim * (self.dim + 1)

        # x = self.proj(x)
        flops += N * self.dim * self.dim
        return flops


class SpatialAttention(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model
        self.proj_1 = nn.Conv2d(d_model, d_model, 1)
        self.activation = nn.GELU()
        self.spatial_gating_unit = AttentionModule(d_model)
        self.proj_2 = nn.Conv2d(d_model, d_model, 1)

    def forward(self, x):
        shorcut = x.clone()
        x = self.proj_1(x)
        x = self.activation(x)
        x = self.spatial_gating_unit(x)
        x = self.proj_2(x)
        x = x + shorcut
        return x


class LKABlock(nn.Module):

    def __init__(self,
                 dim,
                 mlp_ratio=4.,
                 drop=0.,
                 drop_path=0.,
                 act_layer=nn.GELU,
                 linear=False):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)  # build_norm_layer(norm_cfg, dim)[1]
        self.attn = SpatialAttention(dim)
        self.drop_path = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = nn.LayerNorm(dim)  # build_norm_layer(norm_cfg, dim)[1]
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim,
                       act_layer=act_layer, drop=drop, linear=linear)
        layer_scale_init_value = 1e-2
        self.layer_scale_1 = nn.Parameter(
            layer_scale_init_value * torch.ones((dim)), requires_grad=True)
        self.layer_scale_2 = nn.Parameter(
            layer_scale_init_value * torch.ones((dim)), requires_grad=True)

    def forward(self, x):
        y = x.permute(0, 2, 3, 1)  # b h w c, because norm requires this
        y = self.norm1(y)
        y = y.permute(0, 3, 1, 2)  # b c h w, because attn requieres this
        y = self.attn(y)
        y = self.layer_scale_1.unsqueeze(-1).unsqueeze(-1) * y
        y = self.drop_path(y)
        x = x + y
        # x = x + self.drop_path(self.layer_scale_1.unsqueeze(-1).unsqueeze(-1)
        #                       * self.attn(self.norm1(x)))

        y = x.permute(0, 2, 3, 1)  # b h w c, because norm requires this
        y = self.norm2(y)
        y = y.permute(0, 3, 1, 2)  # b c h w, because attn requieres this
        y = self.mlp(y)
        y = self.layer_scale_2.unsqueeze(-1).unsqueeze(-1) * y
        y = self.drop_path(y)
        x = x + y
        # x = x + self.drop_path(self.layer_scale_2.unsqueeze(-1).unsqueeze(-1)
        #                       * self.mlp(self.norm2(x)))
        # x = x.view(B, C, N).permute(0, 2, 1)
        # print("LKA return shape: {}".format(x.shape))
        return x #bchw

class EfficientAttention(nn.Module):
    """
    input  -> x:[B, D, H, W]
    output ->   [B, D, H, W]

    in_channels:    int -> Embedding Dimension
    key_channels:   int -> Key Embedding Dimension,   Best: (in_channels)
    value_channels: int -> Value Embedding Dimension, Best: (in_channels or in_channels//2)
    head_count:     int -> It divides the embedding dimension by the head_count and process each part individually

    Conv2D # of Params:  ((k_h * k_w * C_in) + 1) * C_out)
    """

    def __init__(self, in_channels, key_channels, value_channels, head_count=1):
        super().__init__()
        self.in_channels = in_channels
        self.key_channels = key_channels
        self.head_count = head_count
        self.value_channels = value_channels

        self.keys = nn.Conv2d(in_channels, key_channels, 1)
        self.queries = nn.Conv2d(in_channels, key_channels, 1)
        self.values = nn.Conv2d(in_channels, value_channels, 1)
        self.reprojection = nn.Conv2d(value_channels, in_channels, 1)

    def forward(self, input_):
        n, _, h, w = input_.size()
        ## Here channel weighting and Eigenvalues
        keys = self.keys(input_).reshape((n, self.key_channels, h * w))
        queries = self.queries(input_).reshape(n, self.key_channels, h * w)
        values = self.values(input_).reshape((n, self.value_channels, h * w))

        head_key_channels = self.key_channels // self.head_count
        head_value_channels = self.value_channels // self.head_count

        attended_values = []
        for i in range(self.head_count):
            key = F.softmax(keys[:, i * head_key_channels: (i + 1) * head_key_channels, :], dim=2)

            query = F.softmax(queries[:, i * head_key_channels: (i + 1) * head_key_channels, :], dim=1)

            value = values[:, i * head_value_channels: (i + 1) * head_value_channels, :]

            context = key @ value.transpose(1, 2)  # dk*dv
            attended_value = (context.transpose(1, 2) @ query).reshape(n, head_value_channels, h, w)  # n*dv
            attended_values.append(attended_value)

        aggregated_values = torch.cat(attended_values, dim=1)
        attention = self.reprojection(aggregated_values)

        return attention


class ChannelAttention(nn.Module):
    """
    Input -> x: [B, N, C]
    Output -> [B, N, C]
    """

    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0, proj_drop=0):
        super().__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        """x: [B, N, C]"""
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        q = q.transpose(-2, -1)
        k = k.transpose(-2, -1)
        v = v.transpose(-2, -1)

        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        # -------------------
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).permute(0, 3, 1, 2).reshape(B, N, C)
        # ------------------
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class DualTransformerBlock(nn.Module):
    """
    Input  -> x (Size: (b, (H*W), d)), H, W
    Output -> (b, (H*W), d)
    """

    def __init__(self, in_dim, key_dim, value_dim, head_count=1, token_mlp="mix"):
        super().__init__()
        self.norm1 = nn.LayerNorm(in_dim)
        self.attn = EfficientAttention(in_channels=in_dim, key_channels=key_dim, value_channels=value_dim, head_count=1)
        self.norm2 = nn.LayerNorm(in_dim)
        self.norm3 = nn.LayerNorm(in_dim)
        self.channel_attn = ChannelAttention(in_dim)
        self.norm4 = nn.LayerNorm(in_dim)
        if token_mlp == "mix":
            self.mlp1 = MixFFN(in_dim, int(in_dim * 4))
            self.mlp2 = MixFFN(in_dim, int(in_dim * 4))
        elif token_mlp == "mix_skip":
            self.mlp1 = MixFFN_skip(in_dim, int(in_dim * 4))
            self.mlp2 = MixFFN_skip(in_dim, int(in_dim * 4))
        else:
            self.mlp1 = MLP_FFN(in_dim, int(in_dim * 4))
            self.mlp2 = MLP_FFN(in_dim, int(in_dim * 4))

    def forward(self, x: torch.Tensor, H, W) -> torch.Tensor:
        # dual attention structure, efficient attention first then transpose attention
        norm1 = self.norm1(x)
        norm1 = Rearrange("b (h w) d -> b d h w", h=H, w=W)(norm1)

        attn = self.attn(norm1)
        attn = Rearrange("b d h w -> b (h w) d")(attn)

        add1 = x + attn
        norm2 = self.norm2(add1)
        mlp1 = self.mlp1(norm2, H, W)

        add2 = add1 + mlp1
        norm3 = self.norm3(add2)
        channel_attn = self.channel_attn(norm3)

        add3 = add2 + channel_attn
        norm4 = self.norm4(add3)
        mlp2 = self.mlp2(norm4, H, W)

        mx = add3 + mlp2
        # print("Dual transformer return shape: {}".format(mx.shape))
        return mx


##########################################
#
#         General Decoder Blocks
#
##########################################
class PatchExpand(nn.Module):
    def __init__(self, input_resolution, dim, dim_scale=2, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.expand = nn.Linear(dim, 2 * dim, bias=False) if dim_scale == 2 else nn.Identity()
        self.norm = norm_layer(dim // dim_scale)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        # print("x_shape-----",x.shape)
        H, W = self.input_resolution
        x = self.expand(x)

        B, L, C = x.shape
        # print(x.shape)
        assert L == H * W, "input feature has wrong size"

        x = x.view(B, H, W, C)
        x = rearrange(x, "b h w (p1 p2 c)-> b (h p1) (w p2) c", p1=2, p2=2, c=C // 4)
        x = x.view(B, -1, C // 4)
        x = self.norm(x.clone())

        return x



class FinalPatchExpand_X4(nn.Module):
    def __init__(self, input_resolution, dim, dim_scale=4, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.dim_scale = dim_scale
        self.expand = nn.Linear(dim, 16 * dim, bias=False)
        self.output_dim = dim
        self.norm = norm_layer(self.output_dim)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        x = self.expand(x)
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        x = x.view(B, H, W, C)
        x = rearrange(
            x, "b h w (p1 p2 c)-> b (h p1) (w p2) c", p1=self.dim_scale, p2=self.dim_scale, c=C // (self.dim_scale ** 2)
        )
        x = x.view(B, -1, self.output_dim)
        x = self.norm(x.clone())

        return x
class Conv(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)
class DWConv(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels//2, out_channels//2, kernel_size=kernel_size, stride=stride, padding=padding,groups=out_channels//2),
            nn.BatchNorm2d(out_channels//2),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)
class Out(nn.Module):
    def __init__(self, in_channels, out_channels=9, kernel_size=3, stride=1, padding=1):
        super(Out, self).__init__()
        self.conv1 = Conv(in_channels, in_channels // 4, kernel_size=kernel_size,
                               stride=stride, padding=padding)

        self.conv2 = nn.Conv2d(in_channels // 4, out_channels, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x
##########################################
#
#         MSA^2Net Decoder Blocks
#
##########################################

class MyDecoderLayerLKA(nn.Module):
    def __init__(
            self, input_size, in_out_chan, head_count, token_mlp_mode, reduction_ratio, n_class=1,
            norm_layer=nn.LayerNorm, is_last=False
    ):
        super().__init__()
        dims = in_out_chan[0]
        out_dim = in_out_chan[1]
        key_dim = in_out_chan[2]
        value_dim = in_out_chan[3]
        x1_dim = in_out_chan[4]
        self.input_size = input_size
        reduction_ratio = reduction_ratio
        self.conv = nn.Sequential(
            Conv(in_out_chan[1]*2, in_out_chan[1] // 2),
            Conv(in_out_chan[1] // 2, in_out_chan[1] )

        )
        self.is_last = is_last
        if not is_last:
            self.x2_linear = nn.Linear(x1_dim, out_dim)
            self.ega = EGA(in_channels=x1_dim)

            self.out = Out(in_out_chan[4],n_class)

            self.layer_up = PatchExpand(input_resolution=input_size, dim=out_dim, dim_scale=2, norm_layer=norm_layer)
        else:
            self.x2_linear = nn.Linear(x1_dim, out_dim)
            self.ega = EGA(in_channels=x1_dim)

            self.out = Out(in_out_chan[4], n_class)

            self.layer_up = FinalPatchExpand_X4(
                input_resolution=input_size, dim=out_dim, dim_scale=4, norm_layer=norm_layer
            )

        self.layer_lka_1 = LKABlock(dim=out_dim)
        self.layer_lka_2 = LKABlock(dim=out_dim)
        self.layer_gcn_1 = CFGCN(dim=out_dim)
        self.layer_gcn_2 = CFGCN(dim=out_dim)
        self.dwconv = DWConv(in_channels=dims,out_channels=dims)
        def init_weights(self):
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
                elif isinstance(m, nn.LayerNorm):
                    nn.init.ones_(m.weight)
                    nn.init.zeros_(m.bias)
                elif isinstance(m, nn.Conv2d):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

        init_weights(self)

    def forward(self, x1,edge_feature, x2=None):  # x1是maxvit部分 x1是b,c,h,w x2是b,c,h,w
        H,W = self.input_size
        x1 = x1.contiguous()
        b2,c2,h2,w2=x1.shape
        if x2 is not None:  # skip connection exist
            x2 = x2.contiguous()
            x2_out = self.out(x2)
            ega = self.ega(edge_feature,x1,x2_out)

            cat_linear_x = torch.cat([ega, x1], dim=1)  # B C H W
            out = self.conv(cat_linear_x)
            cat_linear_x = out.contiguous()  # B C H W

            tran_layer_1 = self.layer_lka_1(cat_linear_x)
            # print(tran_layer_1.shape)
            tran_layer_2 = self.layer_lka_2(tran_layer_1) # B C H W

            tran_layer_2 = tran_layer_2.view(tran_layer_2.size(0), tran_layer_2.size(3) * tran_layer_2.size(2),
                                             tran_layer_2.size(1))
            if self.is_last:
                    out = self.layer_up(tran_layer_2).view(b2, 4 * h2, 4 * w2, -1).permute(0, 3, 1, 2)  # 1 9 224 224
            else:
                out = self.layer_up(tran_layer_2).view(b2, 2 * h2, 2 * w2, -1).permute(0, 3, 1, 2) # 1 3136 160
                out = self.dwconv(out)

        else:
            tran_layer_1 = self.layer_lka_1(x1)
            tran_layer_2 = self.layer_lka_2(tran_layer_1)  # B C H W
            tran_layer_2 = tran_layer_2.view(tran_layer_2.size(0), tran_layer_2.size(3) * tran_layer_2.size(2),
                                             tran_layer_2.size(1))
            out = self.layer_up(tran_layer_2).view(b2, 2 * h2, 2 * w2, -1).permute(0, 3, 1, 2)
            out = self.dwconv(out)
        return out



class MyDecoderLayerDAEFormer(nn.Module):
    def __init__(
            self, input_size, in_out_chan, head_count, token_mlp_mode, reduction_ratio, n_class=1,
            norm_layer=nn.LayerNorm, is_last=False
    ):
        super().__init__()
        dims = in_out_chan[0]
        out_dim = in_out_chan[1]
        key_dim = in_out_chan[2]
        value_dim = in_out_chan[3]
        x1_dim = in_out_chan[4]
        reduction_ratio = reduction_ratio
        head_count = head_count
        self.input_size = input_size
        self.conv = nn.Sequential(
            Conv(in_out_chan[1] * 2, in_out_chan[1] // 2),
            Conv(in_out_chan[1] // 2, in_out_chan[1])
        )
        self.is_last = is_last
        # print("Dim: {} | Out_dim: {} | Key_dim: {} | Value_dim: {} | X1_dim: {}".format(dims, out_dim, key_dim, value_dim, x1_dim))
        if not is_last:
            self.ega = EGA(in_channels=x1_dim)
            self.out = Out(in_out_chan[4],n_class)

            self.layer_up = PatchExpand(input_resolution=input_size, dim=out_dim, dim_scale=2, norm_layer=norm_layer)
        else:
            self.ega = EGA(in_channels=x1_dim)

            self.out = Out(in_out_chan[4],n_class)

            self.layer_up = FinalPatchExpand_X4(
                input_resolution=input_size, dim=out_dim, dim_scale=4, norm_layer=norm_layer
            )

        self.tran_layer1 = DualTransformerBlock(in_dim=dims,
                                                key_dim=key_dim,
                                                value_dim=value_dim,
                                                head_count=head_count)
        self.tran_layer2 = DualTransformerBlock(in_dim=dims,
                                                key_dim=key_dim,
                                                value_dim=value_dim,
                                                head_count=head_count)
        self.dwconv = DWConv(in_channels=dims,out_channels=dims)

        def init_weights(self):
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
                elif isinstance(m, nn.LayerNorm):
                    nn.init.ones_(m.weight)
                    nn.init.zeros_(m.bias)
                elif isinstance(m, nn.Conv2d):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

        init_weights(self)

    def forward(self, x1,edge_feature, x2=None):# x1是maxvit部分 x1是b,c,h,w x2是b, h,w, c
        H,W = self.input_size
        x1 = x1.contiguous()
        b2, c2, h2, w2 = x1.shape
        if x2 is not None:  # skip connection exist
            x2 = x2.contiguous()
            x2_out = self.out(x2)
            ega = self.ega(edge_feature, x1, x2_out)
            # cat_linear_x = torch.cat([x2, x1], dim=1)  # B C H W
            cat_linear_x = torch.cat([x1, ega], dim=1)  # B C H W

            out = self.conv(cat_linear_x)
            cat_linear_x = out.contiguous()  # B C H W


            cat_linear_x = cat_linear_x.permute(0, 2, 3, 1).view(b2, -1, c2)  # B H W C --> B (HW) C

            tran_layer_1 = self.tran_layer1(cat_linear_x, h2, w2)  # B N C
            tran_layer_2 = self.tran_layer2(tran_layer_1, h2, w2)  # B N C

            if self.is_last:
                out = self.layer_up(tran_layer_2).view(b2, 4 * h2, 4 * w2, -1).permute(0, 3, 1, 2) # 1 9 224 224
            else:
                out = self.layer_up(tran_layer_2).view(b2, 2 * h2, 2 * w2, -1).permute(0, 3, 1, 2)  # 1 3136 160
                out = self.dwconv(out)

        else:
            x1_expand = x1.view(x1.size(0), x1.size(3) * x1.size(2),x1.size(1))
            tran_layer_1 = self.tran_layer1(x1_expand,H, W )
            tran_layer_2 = self.tran_layer1(tran_layer_1,H, W)  # B N C
            out = self.layer_up(tran_layer_2).view(b2, 2 * h2, 2 * w2, -1).permute(0, 3, 1, 2)
            out = self.dwconv(out)
        return out

##########################################
#
#                MSA^2Net
#
##########################################
# from .merit_lib.networks import MaxViT4Out_Small#, MaxViT4Out_Small3D
# from networks.merit_lib.networks import MaxViT4Out_Small
# from .merit_lib.decoders import CASCADE_Add, CASCADE_Cat

class Msa2Net(nn.Module):
    def __init__(self, num_classes=1, token_mlp_mode="mix_skip"):
        super().__init__()

        # Encoder
        self.backbone = MaxViT4Out_Small(n_class=num_classes, img_size=224,pretrain=True)

        # Decoder
        d_base_feat_size = 7  # 16 for 512 input size, and 7 for 224
        in_out_chan = [
            [96, 96, 96, 96, 96],
            [192, 192, 192, 192, 192],
            [384, 384, 384, 384, 384],
            [768, 768, 768, 768, 768],
        ]  # [dim, out_dim, key_dim, value_dim, x2_dim]
        reduction_ratio = [16, 8, 6, 2]
        head_count = [32, 16, 1, 1]
        self.out1 = Out(96, num_classes)
        self.out2 = Out(96, num_classes)
        self.out3 = Out(192, num_classes)
        self.out4 = Out(384, num_classes)
        self.decoder_3 = MyDecoderLayerDAEFormer(
            (d_base_feat_size, d_base_feat_size),
            in_out_chan[3],
            head_count[0],
            token_mlp_mode,
            n_class=num_classes,
            reduction_ratio=reduction_ratio[0])

        self.decoder_2 = MyDecoderLayerDAEFormer(
            (d_base_feat_size * 2, d_base_feat_size * 2),
            in_out_chan[2],
            head_count[1],
            token_mlp_mode,
            n_class=num_classes,
            reduction_ratio=reduction_ratio[1])
        self.decoder_1 = MyDecoderLayerDAEFormer(
            (d_base_feat_size * 4, d_base_feat_size * 4),
            in_out_chan[1],
            head_count[2],
            token_mlp_mode,
            n_class=num_classes,
            reduction_ratio=reduction_ratio[2])
        self.decoder_0 = MyDecoderLayerDAEFormer(
            (d_base_feat_size * 8, d_base_feat_size * 8),
            in_out_chan[0],
            head_count[3],
            token_mlp_mode,
            n_class=num_classes,
            is_last=True,
            reduction_ratio=reduction_ratio[3])

    def forward(self, x):
        # ---------------Encoder-------------------------
        if x.size()[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        # 将单通道的图像变成具有3个相同通道的图像
        grayscale_img = rgb_to_grayscale(x)
        edge_feature = make_sobel_pyramid(grayscale_img, 5, 1)
        edge_feature = edge_feature[1]

        output_enc_3, output_enc_2, output_enc_1, output_enc_0 = self.backbone(x)

        b, c, _, _ = output_enc_3.shape
        # print(output_enc_3.shape)
        # ---------------Decoder-------------------------

        tmp_3 = self.decoder_3(output_enc_3, edge_feature) #return B N C
        tmp_2 = self.decoder_2(output_enc_2, edge_feature, tmp_3)
        tmp_1 = self.decoder_1(output_enc_1, edge_feature, tmp_2)
        tmp_0 = self.decoder_0(output_enc_0, edge_feature, tmp_1)
        out1 = self.out1(tmp_0)
        out2 = self.out2(tmp_1)
        out3 = self.out3(tmp_2)
        out4 = self.out4(tmp_3)
        return out1,out2,out3,out4

if __name__ == "__main__":
    input0 = torch.rand((1, 3, 224, 224)).cuda(0)
    dec1 = Msa2Net().cuda(0)
    output1 = dec1(input0)
    print("Out shape: " + str(output1.shape) )

    # print(output1)
    # print("Out shape: " + str(output0.shape))
