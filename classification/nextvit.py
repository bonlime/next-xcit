# Copyright (c) ByteDance Inc. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from functools import partial
from timm.models.layers import DropPath, trunc_normal_
from timm.models.registry import register_model

# from utils import merge_pre_bn

from pytorch_tools.modules.residual import FastGlobalAvgPool2d

NORM_EPS = 1e-5


def conv1x1(in_planes, out_planes, stride=1, bias=False):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=bias)


class ConvBNReLU(nn.Sequential):
    residual: bool = False

    def __init__(self, in_channels, out_channels, kernel_size, stride, groups=1):
        super().__init__()

        self.in_chs = in_channels
        self.out_chs = out_channels

        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=1, groups=groups, bias=False
            ),
            nn.BatchNorm2d(out_channels, eps=NORM_EPS),
            nn.ReLU(inplace=True),
        )
        need_pool = (stride > 1) and self.residual
        self.pool = nn.AvgPool2d(stride) if need_pool else nn.Identity()

    def forward(self, x):
        out = self.block(x)
        if self.residual:
            x_ = self.pool(x)
            if self.in_chs < self.out_chs:
                out_1, out_2 = out.split((self.in_chs, self.out_chs - self.in_chs), dim=1)
                out_1 = out_1 + x_
                return torch.cat([out_1, out_2], dim=1)
                # for inference could use inplace version
                # out[:, :self.in_chs] = x_
                # return out
            elif self.in_chs > self.out_chs:
                return out + x_[:, : self.out_chs]
            else:  # in_chs == out_chs
                return out + x_
        else:
            return out


# commenting it out as it increases memory footprint, while not bringing any real improvements
# class LayerScale2d(nn.Module):
#     def __init__(self, dim, init_values=1e-5):
#         super().__init__()
#         self.gamma = nn.Parameter(init_values * torch.ones(1, dim, 1, 1))

#     def forward(self, x):
#         # checkpoint gives -7-10% reduction in memory and -0% in speed
#         return checkpoint(torch.mul, x, self.gamma, use_reentrant=False)
#         # return x * self.gamma


def _make_divisible(v, divisor, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class PatchEmbed(nn.Sequential):
    def __init__(self, in_channels, out_channels, stride=1):
        layers = []
        if stride == 2:
            layers.append(nn.AvgPool2d((2, 2), stride=2))
        if in_channels != out_channels or stride == 2:
            layers.append(conv1x1(in_channels, out_channels))
            layers.append(nn.BatchNorm2d(out_channels, eps=NORM_EPS))
        super().__init__(*layers)


class MHCA(nn.Module):
    """
    Multi-Head Convolutional Attention
    """

    def __init__(self, out_channels, head_dim):
        super(MHCA, self).__init__()
        norm_layer = partial(nn.BatchNorm2d, eps=NORM_EPS)
        self.group_conv3x3 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1, groups=out_channels // head_dim, bias=False
        )
        self.norm = norm_layer(out_channels)
        self.act = nn.ReLU(inplace=True)
        self.projection = nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False)

    def forward(self, x):
        out = self.group_conv3x3(x)
        out = self.norm(out)
        out = self.act(out)
        out = self.projection(out)
        return out


class Mlp(nn.Sequential):
    def __init__(self, in_features, out_features=None, mlp_ratio=None, drop=0.0, bias=True):
        out_features = out_features or in_features
        hidden_dim = _make_divisible(in_features * mlp_ratio, 32)
        layers = [
            nn.BatchNorm2d(in_features, eps=NORM_EPS),
            nn.Conv2d(in_features, hidden_dim, kernel_size=1, bias=bias),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, out_features, kernel_size=1, bias=bias),
            nn.Dropout(drop),
        ]
        super().__init__(*layers)


class NCB(nn.Module):
    """
    Next Convolution Block
    """

    def __init__(self, in_channels, out_channels, stride=1, path_dropout=0.0, drop=0, head_dim=32, mlp_ratio=3):
        super(NCB, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        assert out_channels % head_dim == 0

        if in_channels != out_channels or stride != 1:
            self.patch_embed = PatchEmbed(in_channels, out_channels, stride)
        else:
            self.patch_embed = nn.Identity()
        self.mhca = nn.Sequential(
            MHCA(out_channels, head_dim),
            # LayerScale2d(out_channels),
            DropPath(path_dropout),
        )
        self.mlp = nn.Sequential(
            Mlp(out_channels, mlp_ratio=mlp_ratio, drop=drop, bias=True),
            # LayerScale2d(out_channels),
            DropPath(path_dropout),
        )
        self.is_bn_merged = False

    def forward(self, x):
        x = self.patch_embed(x)
        x = x + self.mhca(x)
        x = x + self.mlp(x)
        return x



class E_MHSA(nn.Module):
    """
    Efficient Multi-Head Self Attention for BCHW input and proper AvgPool
    """
    def __init__(self, dim, out_dim=None, head_dim=32, qkv_bias=True, qk_scale=None,
                 attn_drop=0, proj_drop=0., sr_ratio=1):
        super().__init__()
        self.dim = dim
        self.out_dim = out_dim if out_dim is not None else dim
        self.num_heads = self.dim // head_dim
        self.scale = qk_scale or head_dim ** -0.5
        self.q = nn.Conv2d(dim, self.dim, kernel_size=1, bias=qkv_bias)
        self.k = nn.Conv2d(dim, self.dim, kernel_size=1, bias=qkv_bias)
        self.v = nn.Conv2d(dim, self.dim, kernel_size=1, bias=qkv_bias)
        self.proj = nn.Conv2d(self.dim, self.out_dim, kernel_size=1)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.AvgPool2d(kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.BatchNorm2d(dim, eps=NORM_EPS)
        else:
            self.sr = nn.Identity()
            self.norm = nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        q = self.q(x)
        # -> [B, Hd, C', N] -> [B, Hd, N, C']
        q = q.reshape(B, self.num_heads, C // self.num_heads, -1).transpose(-1, -2)

        x_ = self.sr(x)
        k = self.k(x_)
        # -> [B, Hd, C', N]
        k = k.reshape(B, self.num_heads, C // self.num_heads, -1)
        v = self.v(x_)
        # -> [B, Hd, C', N]
        v = v.reshape(B, self.num_heads, C // self.num_heads, -1)

        # [B, Hd, N, C'] @ [B, Hd, C', Npool] -> [B, Hd, N, Npool]
        attn = (q @ k) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # [B, Hd, C', Npool] @ [B, Hd, Npool, N] -> [B, Hd, C', N]
        x = (v @ attn.transpose(-1, -2)).reshape(B, C, H, W)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class E_MHSA_Original(nn.Module):
    """Using slow  [B N C] format """
    def __init__(self, dim, out_dim=None, head_dim=32, qkv_bias=True, qk_scale=None,
                 attn_drop=0, proj_drop=0., sr_ratio=1):
        super().__init__()
        self.dim = dim
        self.out_dim = out_dim if out_dim is not None else dim
        self.num_heads = self.dim // head_dim
        self.scale = qk_scale or head_dim ** -0.5
        self.q = nn.Linear(dim, self.dim, bias=qkv_bias)
        self.k = nn.Linear(dim, self.dim, bias=qkv_bias)
        self.v = nn.Linear(dim, self.dim, bias=qkv_bias)
        self.proj = nn.Linear(self.dim, self.out_dim)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

        self.sr_ratio = sr_ratio
        self.N_ratio = sr_ratio ** 2
        if sr_ratio > 1:
            self.sr = nn.AvgPool1d(kernel_size=self.N_ratio, stride=self.N_ratio)
            self.norm = nn.BatchNorm1d(dim, eps=NORM_EPS)
        else:
            self.sr = nn.Identity()
            self.norm = nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        N = H * W
        x = rearrange(x, "b c h w -> b (h w) c")  # b n c
        q = self.q(x)
        q = q.reshape(B, N, self.num_heads, int(C // self.num_heads)).permute(0, 2, 1, 3)

        x_ = x.transpose(1, 2)
        x_ = self.sr(x_)
        x_ = self.norm(x_)
        x_ = x_.transpose(1, 2)
        k = self.k(x_)
        k = k.reshape(B, -1, self.num_heads, int(C // self.num_heads)).permute(0, 2, 3, 1)
        v = self.v(x_)
        v = v.reshape(B, -1, self.num_heads, int(C // self.num_heads)).permute(0, 2, 1, 3)
        attn = (q @ k) * self.scale

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        x = rearrange(x, "b (h w) c -> b c h w", h=H)
        return x


class XCA_mod(nn.Module):
    """Cross-Covariance Attention (XCA)
    Operation where the channels are updated using a weighted sum. The weights are obtained from the (softmax
    normalized) Cross-covariance matrix (Q^T \\cdot K \\in d_h \\times d_h)
    This could be viewed as dynamic 1x1 convolution
    """

    def __init__(self, dim: int, head_dim=32, qkv_bias=True, downscale_factor: int = 1):
        super().__init__()
        self.num_heads = dim // head_dim
        self.temperature = nn.Parameter(torch.ones(self.num_heads, 1, 1))
        self.qk = conv1x1(dim, dim * 2, bias=qkv_bias)
        self.v = conv1x1(dim, dim, bias=qkv_bias)
        self.proj = nn.Sequential(conv1x1(dim, dim, bias=True))
        self.downscale_factor = downscale_factor

        if downscale_factor > 1:
            self.down = nn.AvgPool2d(kernel_size=downscale_factor)
        else:
            self.down = nn.Identity()

    def forward(self, x: torch.Tensor):
        B, C, H, W = x.shape
        # C` == channels per head, Hd == num heads
        # -> x B x Hd x C` x N
        v = self.v(x).reshape(B, self.num_heads, C // self.num_heads, -1)
        x_ = self.down(x)
        # -> x B x Hd x C` x N_small
        q, k = self.qk(x_).reshape(B, 2, self.num_heads, C // self.num_heads, -1).unbind(dim=1)

        # Paper section 3.2 l2-Normalization and temperature scaling
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)
        # -> B x Hd x C` x C`
        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        # B x Hd x C` x C` @ B x Hd x C` x H*W -> B x C x H x W
        x_out = (attn @ v).reshape(B, C, H, W)
        x_out = self.proj(x_out)
        return x_out


class NTB(nn.Module):
    """
    Next Transformer Block
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        path_dropout,
        stride=1,
        sr_ratio=1,
        mlp_ratio=2,
        head_dim=32,
        mix_block_ratio=0.75,
        attn_drop=0,
        attn_type: str = "xca",
        drop=0,
    ):
        super(NTB, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.mix_block_ratio = mix_block_ratio

        self.mhsa_out_channels = _make_divisible(int(out_channels * mix_block_ratio), 32)
        self.mhca_out_channels = out_channels - self.mhsa_out_channels

        if attn_type == "xca":
            attn = XCA_mod(self.mhsa_out_channels, head_dim=head_dim, downscale_factor=sr_ratio)
        elif attn_type == "vit":
            attn = E_MHSA(self.mhsa_out_channels, head_dim=head_dim, sr_ratio=sr_ratio)
        elif attn_type == "vit_BNC":
            attn = E_MHSA_Original(self.mhsa_out_channels, head_dim=head_dim, sr_ratio=sr_ratio)
        else:
            raise ValueError(f"attn_type: {attn_type} is not supported")
        self.patch_embed = PatchEmbed(in_channels, self.mhsa_out_channels, stride)
        self.e_mhsa = nn.Sequential(
            nn.BatchNorm2d(self.mhsa_out_channels),
            attn,
            DropPath(path_dropout * mix_block_ratio),
        )

        self.projection = PatchEmbed(self.mhsa_out_channels, self.mhca_out_channels, stride=1)
        self.mhca = nn.Sequential(
            MHCA(self.mhca_out_channels, head_dim=head_dim),
            DropPath(path_dropout * (1 - mix_block_ratio)),
        )

        self.mlp = nn.Sequential(
            Mlp(out_channels, mlp_ratio=mlp_ratio, drop=drop),
            DropPath(path_dropout),
        )

    def forward(self, x):
        x = self.patch_embed(x)
        x = x + self.e_mhsa(x)

        out = self.projection(x)
        out = out + self.mhca(out)
        x = torch.cat([x, out], dim=1)
        x = x + self.mlp(x)
        return x


class NextViT(nn.Module):
    def __init__(
        self,
        stem_chs,
        depths,
        path_dropout,
        attn_drop=0.0,
        drop=0.0,
        num_classes=1000,
        strides=[1, 2, 2, 2],
        sr_ratios=[8, 4, 2, 1],
        head_dim=32,
        mix_block_ratio=0.75,
        attn_type: str = "xca",
    ):
        super(NextViT, self).__init__()
        
        self.stage_out_channels = [
            [96] * (depths[0]),
            [192] * (depths[1] - 1) + [256],
            [384, 384, 384, 384, 512] * (depths[2] // 5),
            [768] * (depths[3] - 1) + [1024],
        ]

        # Next Hybrid Strategy
        self.stage_block_types = [
            [NCB] * depths[0],
            [NCB] * (depths[1] - 1) + [NTB],
            [NCB, NCB, NCB, NCB, NTB] * (depths[2] // 5),
            [NCB] * (depths[3] - 1) + [NTB],
        ]

        self.stem = nn.Sequential(
            ConvBNReLU(3, stem_chs[0], kernel_size=3, stride=2),
            ConvBNReLU(stem_chs[0], stem_chs[1], kernel_size=3, stride=1),
            ConvBNReLU(stem_chs[1], stem_chs[2], kernel_size=3, stride=1),
            ConvBNReLU(stem_chs[2], stem_chs[2], kernel_size=3, stride=2),
        )
        input_channel = stem_chs[-1]
        features = []
        idx = 0
        dpr = [x.item() for x in torch.linspace(0, path_dropout, sum(depths))]  # stochastic depth decay rule
        for stage_id in range(len(depths)):
            numrepeat = depths[stage_id]
            output_channels = self.stage_out_channels[stage_id]
            block_types = self.stage_block_types[stage_id]
            for block_id in range(numrepeat):
                if strides[stage_id] == 2 and block_id == 0:
                    stride = 2
                else:
                    stride = 1
                output_channel = output_channels[block_id]
                block_type = block_types[block_id]
                if block_type is NCB:
                    layer = NCB(
                        input_channel,
                        output_channel,
                        stride=stride,
                        path_dropout=dpr[idx + block_id],
                        drop=drop,
                        head_dim=head_dim,
                    )
                    features.append(layer)
                elif block_type is NTB:
                    layer = NTB(
                        input_channel,
                        output_channel,
                        path_dropout=dpr[idx + block_id],
                        stride=stride,
                        sr_ratio=sr_ratios[stage_id],
                        head_dim=head_dim,
                        mix_block_ratio=mix_block_ratio,
                        attn_drop=attn_drop,
                        attn_type=attn_type,
                        drop=drop,
                    )
                    features.append(layer)
                input_channel = output_channel
            idx += numrepeat
        self.features = nn.Sequential(*features)

        self.norm = nn.BatchNorm2d(output_channel, eps=NORM_EPS)

        # self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.avgpool = FastGlobalAvgPool2d(flatten=True)
        self.proj_head = nn.Sequential(
            nn.Linear(output_channel, num_classes),
        )

        self.stage_out_idx = [sum(depths[: idx + 1]) - 1 for idx in range(len(depths))]
        print("initialize_weights...")
        self._initialize_weights()

    # def merge_bn(self):
    #     self.eval()
    #     for idx, module in self.named_modules():
    #         if isinstance(module, NCB) or isinstance(module, NTB):
    #             module.merge_bn()
    def _initialize_weights(self):
        for n, m in self.named_modules():
            if isinstance(m, (nn.BatchNorm2d, nn.GroupNorm, nn.LayerNorm, nn.BatchNorm1d)):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=0.02)
                if hasattr(m, "bias") and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv2d):
                trunc_normal_(m.weight, std=0.02)
                if hasattr(m, "bias") and m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.stem(x)
        for idx, layer in enumerate(self.features):
            x = layer(x)
        x = self.norm(x)
        x = self.avgpool(x)
        # x = torch.flatten(x, 1)
        x = self.proj_head(x)
        return x


@register_model
def nextvit_small(pretrained=False, pretrained_cfg=None, **kwargs):
    model = NextViT(stem_chs=[64, 32, 64], depths=[3, 4, 10, 3], path_dropout=0.1, **kwargs)
    return model


@register_model
def nextvit_base(pretrained=False, pretrained_cfg=None, **kwargs):
    model = NextViT(stem_chs=[64, 32, 64], depths=[3, 4, 20, 3], path_dropout=0.2, **kwargs)
    return model


@register_model
def nextvit_large(pretrained=False, pretrained_cfg=None, **kwargs):
    model = NextViT(stem_chs=[64, 32, 64], depths=[3, 4, 30, 3], path_dropout=0.2, **kwargs)
    return model
