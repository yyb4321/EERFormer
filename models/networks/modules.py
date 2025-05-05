import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np 

def gauss_kernel(channels=3, cuda=True):
    kernel = torch.tensor([[1., 4., 6., 4., 1],
                            [4., 16., 24., 16., 4.],
                            [6., 24., 36., 24., 6.],
                            [4., 16., 24., 16., 4.],
                            [1., 4., 6., 4., 1.]])
    kernel /= 256.
    kernel = kernel.repeat(channels, 1, 1, 1)
    if cuda:
        kernel = kernel.cuda()
    return kernel

def downsample(x):
    return x[:, :, ::2, ::2]

def conv_gauss(img, kernel):
    img = F.pad(img, (2, 2, 2, 2), mode='reflect')
    out = F.conv2d(img, kernel, groups=img.shape[1])
    return out

def upsample(x, channels):
    cc = torch.cat([x, torch.zeros(x.shape[0], x.shape[1], x.shape[2], x.shape[3], device=x.device)], dim=3)
    cc = cc.view(x.shape[0], x.shape[1], x.shape[2] * 2, x.shape[3])
    cc = cc.permute(0, 1, 3, 2)
    cc = torch.cat([cc, torch.zeros(x.shape[0], x.shape[1], x.shape[3], x.shape[2] * 2, device=x.device)], dim=3)
    cc = cc.view(x.shape[0], x.shape[1], x.shape[3] * 2, x.shape[2] * 2)
    x_up = cc.permute(0, 1, 3, 2)
    return conv_gauss(x_up, 4 * gauss_kernel(channels))

def make_laplace(img, channels):
    filtered = conv_gauss(img, gauss_kernel(channels))
    down = downsample(filtered)
    up = upsample(down, channels)
    if up.shape[2] != img.shape[2] or up.shape[3] != img.shape[3]:
        up = nn.functional.interpolate(up, size=(img.shape[2], img.shape[3]))
    diff = img - up
    return diff

def make_laplace_pyramid(img, level, channels):
    current = img
    pyr = []
    for _ in range(level):
        filtered = conv_gauss(current, gauss_kernel(channels))
        down = downsample(filtered)
        up = upsample(down, channels)
        if up.shape[2] != current.shape[2] or up.shape[3] != current.shape[3]:
            up = nn.functional.interpolate(up, size=(current.shape[2], current.shape[3]))
        diff = current - up
        pyr.append(diff)
        current = down
    pyr.append(current)
    return pyr


class ChannelGate(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16):
        super(ChannelGate, self).__init__()
        self.gate_channels = gate_channels
        self.mlp = nn.Sequential(
            nn.Flatten(),
            nn.Linear(gate_channels, gate_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(gate_channels // reduction_ratio, gate_channels)
            )
    def forward(self, x):
        avg_out = self.mlp(F.avg_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3))))
        max_out = self.mlp(F.max_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3))))
        channel_att_sum = avg_out + max_out

        scale = torch.sigmoid(channel_att_sum).unsqueeze(2).unsqueeze(3).expand_as(x)
        return x * scale

class SpatialGate(nn.Module):
    def __init__(self):
        super(SpatialGate, self).__init__()
        kernel_size = 7
        self.spatial = nn.Conv2d(2, 1, kernel_size, stride=1, padding=(kernel_size-1) // 2)
    def forward(self, x):
        x_compress = torch.cat((torch.max(x,1)[0].unsqueeze(1), torch.mean(x,1).unsqueeze(1)), dim=1)
        x_out = self.spatial(x_compress)
        scale = torch.sigmoid(x_out) # broadcasting
        return x * scale

class CBAM(nn.Module): # 通道注意力和空间注意力串行
    def __init__(self, gate_channels, reduction_ratio=16):
        super(CBAM, self).__init__()
        self.ChannelGate = ChannelGate(gate_channels, reduction_ratio)
        self.SpatialGate = SpatialGate()
    def forward(self, x):
        x_out = self.ChannelGate(x)
        x_out = self.SpatialGate(x_out)
        return x_out

# class ChannelAttentionCBAM(nn.Module):
#     def __init__(self, in_channels, reduction_ratio=16):
#         super(ChannelAttentionCBAM, self).__init__()
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#         self.max_pool = nn.AdaptiveMaxPool2d(1)
#
#         self.fc1 = nn.Conv2d(in_channels, in_channels // reduction_ratio, kernel_size=1, bias=False)
#         self.relu = nn.ReLU(inplace=True)
#         self.fc2 = nn.Conv2d(in_channels // reduction_ratio, in_channels, kernel_size=1, bias=False)
#         self.sigmoid = nn.Sigmoid()
#
#     def forward(self, x):
#         avg_out = self.fc2(self.relu(self.fc1(self.avg_pool(x))))
#         max_out = self.fc2(self.relu(self.fc1(self.max_pool(x))))
#         out = avg_out + max_out
#         return self.sigmoid(out)
#
#
# # Squeeze-and-excitation
# class SEBlock(nn.Module):
#     def __init__(self, in_channels, reduction_ratio=16):
#         super(SEBlock, self).__init__()
#         self.squeeze = nn.AdaptiveAvgPool2d(1)
#         self.excitation = nn.Sequential(
#             nn.Linear(in_channels, in_channels // reduction_ratio),
#             nn.ReLU(inplace=True),
#             nn.Linear(in_channels // reduction_ratio, in_channels),
#             nn.Sigmoid()
#         )
#
#     def forward(self, x):
#         batch_size, channels, _, _ = x.size()
#         y = self.squeeze(x).view(batch_size, channels)
#         y = self.excitation(y).view(batch_size, channels, 1, 1)
#         return x * y
#
#
# class SpatialAttentionCBAM(nn.Module):
#     def __init__(self, kernel_size=7):
#         super(SpatialAttentionCBAM, self).__init__()
#
#         assert kernel_size in (3, 7), "kernel size must be 3 or 7"
#         padding = 3 if kernel_size == 7 else 1
#
#         self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
#         self.sigmoid = nn.Sigmoid()
#
#     def forward(self, x):
#         avg_out = torch.mean(x, dim=1, keepdim=True)
#         max_out, _ = torch.max(x, dim=1, keepdim=True)
#         out = torch.cat([avg_out, max_out], dim=1)
#         out = self.conv(out)
#         return self.sigmoid(out)
#
# # 通道注意力和空间注意力并行
# class CBAM(nn.Module):
#     def __init__(self, in_channels, reduction_ratio=16, kernel_size=7):
#         super(CBAM, self).__init__()
#         self.channel_attention = ChannelAttentionCBAM(in_channels, reduction_ratio)
#         self.SE = SEBlock(in_channels, reduction_ratio=16)
#         self.spatial_attention = SpatialAttentionCBAM(kernel_size)
#
#     def forward(self, x):
#         se_out = x * self.channel_attention(x)
#         sa_out = x * self.spatial_attention(x)
#         return torch.add(se_out, sa_out)

class Dilated_Conv(nn.Module):
    """
    Wide-Focus module.
    """

    def __init__(self,
                 in_channels,
                 out_channels):
        super().__init__()
        self.conv1=  nn.Conv2d(in_channels, out_channels, 3, 1, padding="same")
        self.conv2 = nn.Conv2d(in_channels, out_channels, 3, 1, padding="same", dilation=2)
        self.conv3 = nn.Conv2d(in_channels, out_channels, 3, 1, padding="same", dilation=3)
        self.conv4 = nn.Conv2d(in_channels, out_channels, 3, 1, padding="same")

    def forward(self, x):
        x1 = self.conv1(x)
        x1 = F.gelu(x1)
        x1 = F.dropout(x1, 0.1)
        x2 = self.conv2(x)
        x2 = F.gelu(x2)
        x2 = F.dropout(x2, 0.1)
        x3=self.conv3(x)
        x3=F.gelu(x3)
        x3=F.dropout(x3, 0.1)
        added = torch.add(x1, x2)
        added = torch.add(added, x3)
        x_out = self.conv4(added)
        x_out = F.gelu(x_out)
        x_out = F.dropout(x_out, 0.1)
        return x_out

class Weight filtering(nn.Module):
    def __init__(self, dim, Weight_window, Weight_level, Weight_factor=2, bias=True, proj_drop=0.,
                 use_postln_in_modulation=False, normalize_modulator=False):
        super().__init__()

        self.dim = dim
        self.Weight_window = Weight_window
        self.Weight_level = Weight_level
        self.Weight_factor = Weight_factor
        self.use_postln_in_modulation = use_postln_in_modulation
        self.normalize_modulator = normalize_modulator

        self.f = nn.Linear(dim, 2 * dim + (self.Weight_level + 1), bias=bias)
        self.h = nn.Conv2d(dim, dim, kernel_size=1, stride=1, bias=bias)

        self.act = nn.GELU()
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.Weight_layers = nn.ModuleList()

        self.kernel_sizes = []
        for k in range(self.Weight_level):
            kernel_size = self.Weight_factor * k + self.Weight_window
            self.Weightlayers.append(
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
        q, ctx, self.gates = torch.split(x, (C, C, self.Weight_level + 1), 1)

        # context aggreation
        ctx_all = 0
        for l in range(self.Weight_level):
            ctx = self.Weight_layers[l](ctx)
            ctx_all = ctx_all + ctx * self.gates[:, l:l + 1]
        ctx_global = self.act(ctx.mean(2, keepdim=True).mean(3, keepdim=True))
        ctx_all = ctx_all + ctx_global * self.gates[:, self.Weight_level:]

        # normalize context
        if self.normalize_modulator:
            ctx_all = ctx_all / (self.Weight_level + 1)

        # Weight modulation
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

        flops += N * self.dim * (self.dim * 2 + (self.Weight_level + 1))

        # Weight convolution
        for k in range(self.Weight_level):
            flops += N * (self.kernel_sizes[k] ** 2 + 1) * self.dim

        # global gating
        flops += N * 1 * self.dim

        #  self.linear
        flops += N * self.dim * (self.dim + 1)

        # x = self.proj(x)
        flops += N * self.dim * self.dim
        return flops

#Edge-Guided Attention Module
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

    def __init__(self, in_channels, key_channels, value_channels, head_count=4):
        super().__init__()
        self.in_channels = in_channels
        self.key_channels = key_channels
        self.head_count = head_count
        self.value_channels = value_channels

        self.keys = nn.Conv2d(in_channels, key_channels, 1)
        self.queries = nn.Conv2d(in_channels, key_channels, 1)
        self.values = nn.Conv2d(in_channels, value_channels, 1)
        self.reprojection = nn.Conv2d(value_channels, in_channels, 1)

    def forward(self, q,k,v):
        n, _, h, w = q.shape
        ## Here channel weighting and Eigenvalues
        keys = self.keys(k).reshape((n, self.key_channels, h * w))
        queries = self.queries(k).reshape(n, self.key_channels, h * w)
        values = self.values(v).reshape((n, self.value_channels, h * w))

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
class EGA(nn.Module):
    def __init__(self, in_channels):
        super(EGA, self).__init__()

        self.fusion_conv = nn.Sequential(
            nn.Conv2d(in_channels , in_channels, 3 , 1, 1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True))
        self.attention1 = EfficientAttention(in_channels,in_channels,in_channels)
        self.attention = nn.Sequential(
            nn.Conv2d(in_channels, 1, 3, 1, 1),
            nn.BatchNorm2d(1),
            nn.Sigmoid())
        self.conv1 = nn.Conv2d(in_channels, in_channels, 3, 1, padding="same")

        self.layernorm = nn.LayerNorm(self.conv1.in_channels, eps=1e-5)

        self.wide_focus = Dilated_Conv(in_channels, in_channels)
        self.cbam = Weight filtering(in_channels,Weight_window=3, Weight_level=3)

    def forward(self, edge_feature, x, pred):
        residual = x
        xsize = x.size()[2:]
        pred = torch.sigmoid(pred)
        B,C,H,W = x.shape
        #reverse attention 
        background_att = 1 - pred
        background_x= x * background_att
        
        #boudary attention
        edge_pred = make_laplace(pred, 1)
        pred_feature = x * edge_pred

        #high-frequency feature
        edge_input = F.interpolate(edge_feature, size=xsize, mode='bilinear', align_corners=True)
        input_feature = x * edge_input
        fusion_feature = self.attention1(background_x, pred_feature, input_feature)
        fusion_feature = self.fusion_conv(fusion_feature)



        # attention_map = self.attention(fusion_feature)
        # fusion_feature = fusion_feature * attention_map

        out = fusion_feature + residual
        # out1 = out.permute(0, 2, 3, 1)
        # x3 = self.layernorm(out1)
        # x3 = x3.permute(0, 3, 1, 2)
        # x3 = self.wide_focus(x3)
        # x3 = torch.add(out, x3)
        # x4 = self.cbam(out)
        # out = torch.add(out, x4)
        x2 = out.permute(0,2,3,1)
        x4 = self.cbam(x2).permute(0,3,1,2)
        out = torch.add(out, x4)
        # out = self.cbam(out) # 串行时用的cbam

        return out
#
# if __name__ == "__main__":
#     input0 = torch.rand((1, 1, 224, 224)).cuda(0)
#     input1 = torch.rand((1, 1, 224, 224)).cuda(0)
#     input2 = torch.rand((1, 1, 224, 224)).cuda(0)
#     dec1 = EGA(1).cuda(0)
#     output1 = dec1(input0,input1,input2)
#     print("Out shape: " + str(output1.shape) )
