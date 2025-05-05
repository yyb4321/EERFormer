import torch
import torch.nn as nn
from einops import rearrange
from einops.layers.torch import Rearrange
from torch.nn import functional as F

# from .networks.segformer import *

from .segformer import *
from .masag import MultiScaleGatedAttn
from .merit_lib.networks import MaxViT4Out_Small
# from segformer import *
# from attentions import MultiScaleGatedAttn

from timm.models.layers import DropPath, to_2tuple
import math

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
        # B, N, C = x.shape
        # x = x.permute(0, 2, 1).view(B, C, H, W)
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
        return x

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

##########################################
#
#         MSA^2Net Decoder Blocks
#
##########################################

class MyDecoderLayerLKA(nn.Module):
    def __init__(
            self, input_size, in_out_chan, head_count, token_mlp_mode, reduction_ratio, n_class=9,
            norm_layer=nn.LayerNorm, is_last=False
    ):
        super().__init__()
        dims = in_out_chan[0]
        out_dim = in_out_chan[1]
        key_dim = in_out_chan[2]
        value_dim = in_out_chan[3]
        x1_dim = in_out_chan[4]
        reduction_ratio = reduction_ratio
        # print("Dim: {} | Out_dim: {} | Key_dim: {} | Value_dim: {} | X1_dim: {}".format(dims, out_dim, key_dim, value_dim, x1_dim))
        if not is_last:
            self.x1_linear = nn.Linear(x1_dim, out_dim)
            # self.ag_attn = Gated_Attention_block(x1_dim, x1_dim, x1_dim)
            self.ag_attn = MultiScaleGatedAttn(dim=x1_dim)
            self.ag_attn_norm = nn.LayerNorm(out_dim)

            self.layer_up = PatchExpand(input_resolution=input_size, dim=out_dim, dim_scale=2, norm_layer=norm_layer)
            self.last_layer = None
        else:
            self.x1_linear = nn.Linear(x1_dim, out_dim)
            # self.ag_attn = Gated_Attention_block(x1_dim, x1_dim, x1_dim)
            self.ag_attn = MultiScaleGatedAttn(dim=x1_dim)
            self.ag_attn_norm = nn.LayerNorm(out_dim)

            self.layer_up = FinalPatchExpand_X4(
                input_resolution=input_size, dim=out_dim, dim_scale=4, norm_layer=norm_layer
            )
            self.last_layer = nn.Conv2d(out_dim, n_class, 1)

        # self.layer_lka_1 = SinaAttnBlock(dim = out_dim)
        self.layer_lka_1 = LKABlock(dim=out_dim)
        self.layer_lka_2 = LKABlock(dim=out_dim)

        # self.layer_lka_2 = SinaAttnBlock(dim = out_dim)

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

    def forward(self, x1, x2=None):
        if x2 is not None:  # skip connection exist
            x2 = x2.contiguous()
            # b, c, h, w = x1.shape
            b2, h2, w2, c2 = x2.shape  # 1 28 28 320, 1 56 56 128
            x2 = x2.view(b2, -1, c2)  # 1 784 320, 1 3136 128

            x1_expand = self.x1_linear(x1)  # 1 784 256 --> 1 784 320, 1 3136 160 --> 1 3136 128
            #x1_expand = x1
            x2_new = x2.view(x2.size(0), x2.size(2), x2.size(1) // w2, x2.size(1) // h2)

            x1_expand = x1_expand.view(x2.size(0), x2.size(2), x2.size(1) // w2, x2.size(1) // h2)

            # print(f'the x1_expand shape is: {x1_expand.shape}\n\t the x2_new shape is: {x2_new.shape}')

            attn_gate = self.ag_attn(x=x2_new, g=x1_expand)  # B C H W

            cat_linear_x = x1_expand + attn_gate  # B C H W
            cat_linear_x = cat_linear_x.permute(0, 2, 3, 1)  # B H W C
            cat_linear_x = self.ag_attn_norm(cat_linear_x)  # B H W C

            cat_linear_x = cat_linear_x.permute(0, 3, 1, 2).contiguous()  # B C H W

            tran_layer_1 = self.layer_lka_1(cat_linear_x)
            # print(tran_layer_1.shape)
            tran_layer_2 = self.layer_lka_2(tran_layer_1)

            tran_layer_2 = tran_layer_2.view(tran_layer_2.size(0), tran_layer_2.size(3) * tran_layer_2.size(2),
                                             tran_layer_2.size(1))
            if self.last_layer:
                out = self.last_layer(
                    self.layer_up(tran_layer_2).view(b2, 4 * h2, 4 * w2, -1).permute(0, 3, 1, 2))  # 1 9 224 224
            else:
                out = self.layer_up(tran_layer_2)  # 1 3136 160
        else:
            out = self.layer_up(x1)
        return out


class MyDecoderLayerDAEFormer(nn.Module):
    def __init__(
            self, input_size, in_out_chan, head_count, token_mlp_mode, reduction_ratio, n_class=9,
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
        # print("Dim: {} | Out_dim: {} | Key_dim: {} | Value_dim: {} | X1_dim: {}".format(dims, out_dim, key_dim, value_dim, x1_dim))
        if not is_last:
            self.x1_linear = nn.Linear(x1_dim, out_dim)

            self.ag_attn = MultiScaleGatedAttn(dim=x1_dim)
            self.ag_attn_norm = nn.LayerNorm(out_dim)

            self.layer_up = PatchExpand(input_resolution=input_size, dim=out_dim, dim_scale=2, norm_layer=norm_layer)
            self.last_layer = None
        else:
            self.x1_linear = nn.Linear(x1_dim, out_dim)

            self.ag_attn = MultiScaleGatedAttn(dim=x1_dim)
            self.ag_attn_norm = nn.LayerNorm(out_dim)

            self.layer_up = FinalPatchExpand_X4(
                input_resolution=input_size, dim=out_dim, dim_scale=4, norm_layer=norm_layer
            )
            self.last_layer = nn.Conv2d(out_dim, n_class, 1)

        self.tran_layer1 = DualTransformerBlock(in_dim=dims,
                                                key_dim=key_dim,
                                                value_dim=value_dim,
                                                head_count=head_count)
        self.tran_layer2 = DualTransformerBlock(in_dim=dims,
                                                key_dim=key_dim,
                                                value_dim=value_dim,
                                                head_count=head_count)

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

    def forward(self, x1, x2=None):
        if x2 is not None:  # skip connection exist
            x2 = x2.contiguous()

            b2, h2, w2, c2 = x2.shape  # B C H W --> B H W C
            x2 = x2.view(b2, -1, c2)  # B C H W --> B (HW) C

            x1_expand = self.x1_linear(x1)  # B N C
            x2_new = x2.view(x2.size(0), x2.size(2), x2.size(1) // w2, x2.size(1) // h2)  # B (HW) C --> B C H W

            x1_expand = x1_expand.view(x2.size(0), x2.size(2), x2.size(1) // w2, x2.size(1) // h2)  # B N C --> B C H W

            attn_gate = self.ag_attn(x=x2_new, g=x1_expand)  # B C H W

            cat_linear_x = x1_expand + attn_gate  # B C H W
            cat_linear_x = cat_linear_x.permute(0, 2, 3, 1)  # B H W C
            cat_linear_x = self.ag_attn_norm(cat_linear_x)  # B H W C

            # cat_linear_x = cat_linear_x.permute(0, 3, 1, 2).contiguous()  # B C H W
            cat_linear_x = cat_linear_x.view(b2, -1, c2)  # B H W C --> B (HW) C

            tran_layer_1 = self.tran_layer1(cat_linear_x, h2, w2)  # B N C
            # print(tran_layer_1.shape)
            tran_layer_2 = self.tran_layer2(tran_layer_1, h2, w2)  # B N C
            # print(tran_layer_2.shape)

            if self.last_layer:
                out = self.last_layer(
                    self.layer_up(tran_layer_2).view(b2, 4 * h2, 4 * w2, -1).permute(0, 3, 1, 2))  # 1 9 224 224
            else:
                out = self.layer_up(tran_layer_2)  # 1 3136 160
        else:
            out = self.layer_up(x1)
        return out


    def __init__(
            self, input_size, in_out_chan, head_count, token_mlp_mode, reduction_ratio, n_class=9,
            norm_layer=nn.LayerNorm, is_last=False
    ):
        super().__init__()
        dims = in_out_chan[0]
        out_dim = in_out_chan[1]
        key_dim = in_out_chan[2]
        value_dim = in_out_chan[3]
        x1_dim = in_out_chan[4]
        reduction_ratio = reduction_ratio
        # print("Dim: {} | Out_dim: {} | Key_dim: {} | Value_dim: {} | X1_dim: {}".format(dims, out_dim, key_dim, value_dim, x1_dim))
        if not is_last:
            self.x1_linear = nn.Linear(x1_dim, out_dim)
            # self.ag_attn = Gated_Attention_block(x1_dim, x1_dim, x1_dim)
            self.ag_attn = MultiScaleGatedAttn(dim=x1_dim)
            self.ag_attn_norm = nn.LayerNorm(out_dim)

            self.layer_up = PatchExpand(input_resolution=input_size, dim=out_dim, dim_scale=2, norm_layer=norm_layer)
            self.last_layer = None
        else:
            self.x1_linear = nn.Linear(x1_dim, out_dim)
            # self.ag_attn = Gated_Attention_block(x1_dim, x1_dim, x1_dim)
            self.ag_attn = MultiScaleGatedAttn(dim=x1_dim)
            self.ag_attn_norm = nn.LayerNorm(out_dim)

            self.layer_up = FinalPatchExpand_X4(
                input_resolution=input_size, dim=out_dim, dim_scale=4, norm_layer=norm_layer
            )
            self.last_layer = nn.Conv2d(out_dim, n_class, 1)

        # self.layer_lka_1 = SinaAttnBlock(dim = out_dim)
        self.layer_lka_1 = LKABlock(dim=out_dim)
        self.layer_lka_2 = LKABlock(dim=out_dim)

        # self.layer_lka_2 = SinaAttnBlock(dim = out_dim)

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

    def forward(self, x1, x2=None):
        if x2 is not None:  # skip connection exist
            x2 = x2.contiguous()
            # b, c, h, w = x1.shape
            b2, h2, w2, c2 = x2.shape  # 1 28 28 320, 1 56 56 128
            x2 = x2.view(b2, -1, c2)  # 1 784 320, 1 3136 128

            x1_expand = self.x1_linear(x1)  # 1 784 256 --> 1 784 320, 1 3136 160 --> 1 3136 128
            #x1_expand = x1
            x2_new = x2.view(x2.size(0), x2.size(2), x2.size(1) // w2, x2.size(1) // h2)

            x1_expand = x1_expand.view(x2.size(0), x2.size(2), x2.size(1) // w2, x2.size(1) // h2)

            # print(f'the x1_expand shape is: {x1_expand.shape}\n\t the x2_new shape is: {x2_new.shape}')

            # attn_gate = self.ag_attn(x=x2_new, g=x1_expand)  # B C H W

            cat_linear_x = x1_expand + x2_new  # B C H W
            cat_linear_x = cat_linear_x.permute(0, 2, 3, 1)  # B H W C
            cat_linear_x = self.ag_attn_norm(cat_linear_x)  # B H W C

            cat_linear_x = cat_linear_x.permute(0, 3, 1, 2).contiguous()  # B C H W

            tran_layer_1 = self.layer_lka_1(cat_linear_x)
            # print(tran_layer_1.shape)
            tran_layer_2 = self.layer_lka_2(tran_layer_1)

            tran_layer_2 = tran_layer_2.view(tran_layer_2.size(0), tran_layer_2.size(3) * tran_layer_2.size(2),
                                             tran_layer_2.size(1))
            if self.last_layer:
                out = self.last_layer(
                    self.layer_up(tran_layer_2).view(b2, 4 * h2, 4 * w2, -1).permute(0, 3, 1, 2))  # 1 9 224 224
            else:
                out = self.layer_up(tran_layer_2)  # 1 3136 160
        else:
            out = self.layer_up(x1)
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
    def __init__(self, num_classes=9, token_mlp_mode="mix_skip"):
        super().__init__()

        # Encoder
        self.backbone = MaxViT4Out_Small(n_class=num_classes, img_size=224)

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
        self.decoder_1 = MyDecoderLayerLKA(
            (d_base_feat_size * 4, d_base_feat_size * 4),
            in_out_chan[1],
            head_count[2],
            token_mlp_mode,
            n_class=num_classes,
            reduction_ratio=reduction_ratio[2])
        self.decoder_0 = MyDecoderLayerLKA(
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

        output_enc_3, output_enc_2, output_enc_1, output_enc_0 = self.backbone(x)

        b, c, _, _ = output_enc_3.shape
        # print(output_enc_3.shape)
        # ---------------Decoder-------------------------
        tmp_3 = self.decoder_3(output_enc_3.permute(0, 2, 3, 1).view(b, -1, c))
        tmp_2 = self.decoder_2(tmp_3, output_enc_2.permute(0, 2, 3, 1))
        tmp_1 = self.decoder_1(tmp_2, output_enc_1.permute(0, 2, 3, 1))
        tmp_0 = self.decoder_0(tmp_1, output_enc_0.permute(0, 2, 3, 1))

        return tmp_0


if __name__ == "__main__":
    input0 = torch.rand((1, 3, 224, 224)).cuda(0)
    input = torch.randn((1, 768, 7, 7)).cuda(0)
    input2 = torch.randn((1, 384, 14, 14)).cuda(0)
    input3 = torch.randn((1, 192, 28, 28)).cuda(0)
    b, c, _, _ = input.shape
    dec1 = MyDecoderLayerDAEFormer(input_size=(7, 7), in_out_chan=([768, 768, 768, 768, 768]), head_count=32,
                                     token_mlp_mode='mix_skip', reduction_ratio=16).cuda(0)
    dec2 = MyDecoderLayerDAEFormer(input_size=(14, 14), in_out_chan=([384, 384, 384, 384, 384]), head_count=16,
                                     token_mlp_mode='mix_skip', reduction_ratio=8).cuda(0)
    dec3 = MyDecoderLayerLKA(input_size=(28, 28), in_out_chan=([192, 192, 192, 192, 192]), head_count=1,
                                     token_mlp_mode='mix_skip', reduction_ratio=6).cuda(0)
    output = dec1(input.permute(0, 2, 3, 1).view(b, -1, c))
    output2 = dec2(output, input2.permute(0, 2, 3, 1))
    output3 = dec3(output2, input3.permute(0, 2, 3, 1))

    # net = MaxViT_deformableLKAFormer().cuda(0)

    # output0 = net(input0)
    print("Out shape: " + str(output.shape) + 'Out2 shape:' + str(output2.shape) + "Out shape3: " + str(output3.shape))
    # print("Out shape: " + str(output0.shape))