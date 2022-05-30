import torch
import itertools
import numpy as np
from torch import nn, einsum
from torch.nn import functional as F

from .common import conv1x1_bn, conv_bn, sep_bn
from einops import rearrange, repeat
from einops.layers.torch import Rearrange


class ASPPModule(nn.Module):
    """ASPP module of DeepLab V3+. Using separable atrous conv.
    Currently no GAP. Don't think GAP is useful for cityscapes.
    """

    def __init__(self, inp, oup, rates, affine=True, use_gap=True, activate_f='ReLU'):
        super(ASPPModule, self).__init__()
        self.conv1 = conv1x1_bn(inp, oup, affine=affine, activate_f=activate_f)
        self.atrous = nn.ModuleList()
        self.use_gap = use_gap
        for rate in rates:
            self.atrous.append(sep_bn(inp, oup, rate, activate_f=activate_f))
        num_branches = 1 + len(rates)
        if use_gap:
            self.gap = nn.Sequential(nn.AdaptiveAvgPool2d(1),
                                     conv1x1_bn(inp, oup, activate_f=activate_f))
            num_branches += 1
        self.conv_last = conv1x1_bn(oup * num_branches,
                                    oup, affine=affine, activate_f=activate_f)

    def forward(self, x):
        atrous_outs = [atrous(x) for atrous in self.atrous]
        atrous_outs.append(self.conv1(x))
        if self.use_gap:
            gap = self.gap(x)
            gap = F.interpolate(gap, size=x.size()[2:],
                                mode='bilinear', align_corners=False)
            atrous_outs.append(gap)
        x = torch.cat(atrous_outs, dim=1)
        x = self.conv_last(x)
        return x


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


# Transformer module used in VIT
class Transformer_vit(nn.Module):
    def __init__(self, Dim_in, Crop_size, dim_head=32, heads=4, dropout = 0.1, emb_dropout=0.1):
        super(Transformer_vit, self).__init__()
        tokens_num = Crop_size * Crop_size
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.pos_embedding = nn.Parameter(torch.randn(1, tokens_num, Dim_in))
        self.drop_out = nn.Dropout(emb_dropout)
        inner_dim = dim_head*heads
        # attention
        self.attend = torch.nn.Softmax(dim = -1)
        self.to_qkv = nn.Linear(Dim_in, inner_dim*3, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, Dim_in),
            nn.Dropout(dropout)
        )
        # FF
        self.ffnet = nn.Sequential(
            nn.Linear(Dim_in, dim_head*heads),
            nn.GELU(),
            # nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_head*heads, Dim_in),
            nn.Dropout(dropout)
        )
        self.norm1= nn.LayerNorm(Dim_in)
        self.norm2 = nn.LayerNorm(Dim_in)


    def forward(self, x):
        b, c, h, w = x.shape
        sque_x = x.view(b, c, h*w).permute(0, 2, 1)
        x = sque_x + self.pos_embedding
        x = self.drop_out(x)
        qkv = self.to_qkv(self.norm1(x)).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        attn = self.attend(dots)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')

        x = self.to_out(out) + x
        x = self.ffnet(self.norm2(x)) + x

        out = x.permute(0, 2, 1).view(b, c, h, w)
        return out


class Residual(torch.nn.Module):
    def __init__(self, m, drop):
        super().__init__()
        self.m = m
        self.drop = drop

    def forward(self, x):
        if self.training and self.drop > 0:
            return x + self.m(x) * torch.rand(x.size(0), 1, 1,
                                              device=x.device).ge_(self.drop).div(1 - self.drop).detach()
        else:
            return x + self.m(x)


class Linear_BN(torch.nn.Sequential):
    def __init__(self, a, b, bn_weight_init=1):
        super().__init__()
        self.add_module('c', torch.nn.Linear(a, b, bias=False))
        bn = torch.nn.BatchNorm1d(b)
        torch.nn.init.constant_(bn.weight, bn_weight_init)
        torch.nn.init.constant_(bn.bias, 0)
        self.add_module('bn', bn)

    def forward(self, x):
        l, bn = self._modules.values()
        x = l(x)
        return bn(x.flatten(0, 1)).reshape_as(x)


# Transformer module used in LeViT
class Transformer_levit(nn.Module):
    def __init__(self, Dim_in, Crop_size, dim_head=32, heads=4, atn_raito=2, mlp_ratio=2, drop_p = 0.1):
        super(Transformer_levit, self).__init__()

        #attention part
        activation = torch.nn.Hardtanh
        # activation = torch.nn.Hardswish

        self.key_dim = dim_head
        self.num_heads = heads
        self.scale = self.key_dim ** -0.5
        self.nh_kq = self.key_dim * self.num_heads
        self.v_dim = self.key_dim * atn_raito
        self.nh_v = self.key_dim * self.num_heads * atn_raito
        dim_qkv = self.nh_kq * 2 + self.nh_v
        self.qkv = Linear_BN(Dim_in, dim_qkv)
        self.proj = torch.nn.Sequential(activation(),
                                        Linear_BN(self.nh_v, Dim_in, bn_weight_init=0)

        )

        points = list(itertools.product(range(Crop_size), range(Crop_size)))
        N = len(points)
        attention_offsets = {}
        idxs = []
        for p1 in points:
            for p2 in points:
                offset = (abs(p1[0] - p2[0]), abs(p1[1] - p2[1]))
                if offset not in attention_offsets:
                    attention_offsets[offset] = len(attention_offsets)
                idxs.append(attention_offsets[offset])
        self.attention_biases = torch.nn.Parameter(
            torch.zeros(self.num_heads, len(attention_offsets)))
        self.register_buffer('attention_bias_idxs',
                             torch.LongTensor(idxs).view(N, N))

        # x2 MLP

        self.ffnet=Residual(torch.nn.Sequential(
                            Linear_BN(Dim_in, Dim_in * mlp_ratio),
                            activation(),
                            Linear_BN(Dim_in * mlp_ratio, Dim_in, bn_weight_init=0),
                        ), drop_p)


    def forward(self, x):
        B, C, H, W = x.shape
        x_in = x.view(B, C, H*W).permute(0, 2, 1) # x_in (B, N, C)
        qkv = self.qkv(x_in)
        q, k, v = qkv.view(B, H*W, self.num_heads, -1).split([self.key_dim, self.key_dim, self.v_dim], dim=3)

        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)

        attn = (
                (q @ k.transpose(-2, -1)) * self.scale
                +
                self.attention_biases[:, self.attention_bias_idxs]

        )
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B, H*W, self.nh_v)
        x = self.proj(x)

        x = x_in + x
        x = self.ffnet(x)

        out = x.permute(0, 2, 1).view(B, C, H, W)
        return out


# Patch size free Transformer
class Transformer_psf(nn.Module):
    def __init__(self, Dim_in, Crop_size=9, dim_head=32, heads=4, atn_raito=2, mlp_ratio=2, drop_p = 0):
        super(Transformer_psf, self).__init__()

        #attention part
        activation = torch.nn.Hardtanh
        # activation = torch.nn.Hardswish
        self.crop_size = Crop_size
        self.key_dim = dim_head
        self.num_heads = heads
        self.scale = self.key_dim ** -0.5
        self.nh_kq = self.key_dim * self.num_heads
        self.v_dim = self.key_dim * atn_raito
        self.nh_v = self.key_dim * self.num_heads * atn_raito
        dim_qkv = self.nh_kq * 2 + self.nh_v
        self.get_q = torch.nn.Sequential(nn.Conv2d(Dim_in, self.nh_kq, 1),
                                         nn.LayerNorm(self.nh_kq))
        self.get_k = torch.nn.Sequential(nn.Conv2d(Dim_in, self.nh_kq, 1),
                                         nn.LayerNorm(self.nh_kq))
        self.k_pad = torch.nn.ZeroPad2d(self.crop_size // 2)
        self.get_v = torch.nn.Sequential(nn.Conv2d(Dim_in, self.nh_v, 1),
                                         nn.LayerNorm(self.nh_v))

        self.proj = torch.nn.Sequential(activation(),
                                        nn.Conv2d(self.nh_v, Dim_in, 1),
                                        nn.LayerNorm(Dim_in)
                                        )

        self.r_pos_bias = torch.nn.Parameter(
            torch.zeros(self.num_heads, self.crop_size, self.crop_size))


        points = list(itertools.product(range(Crop_size), range(Crop_size)))
        N = len(points)
        attention_offsets = {}
        idxs = []
        for p1 in points:
            for p2 in points:
                offset = (abs(p1[0] - p2[0]), abs(p1[1] - p2[1]))
                if offset not in attention_offsets:
                    attention_offsets[offset] = len(attention_offsets)
                idxs.append(attention_offsets[offset])
        self.attention_biases = torch.nn.Parameter(
            torch.zeros(self.num_heads, len(attention_offsets)))
        self.register_buffer('attention_bias_idxs',
                             torch.LongTensor(idxs).view(N, N))

        # x2 MLP

        self.ffnet=Residual(torch.nn.Sequential(
                            Linear_BN(Dim_in, Dim_in * mlp_ratio),
                            activation(),
                            Linear_BN(Dim_in * mlp_ratio, Dim_in, bn_weight_init=0),
                        ), drop_p)


    def forward(self, x):
        B, C, H, W = x.shape
        k = self.k_pad(self.get_k(x))
        q = self.get_q(x)
        v = self.get_v(x)
        atten = []
        for i in range(self.crop_size):
            for j in range(self.crop_size):
                atten_patch = torch.sum(k[:, :, i:i+H, j:j+W]*q, dim=2)*self.scale + self.r_pos_bias[:, i, j]
                atten.append(atten_patch)

        return atten


class Decoder(nn.Module):
    """DeepLab V3+ decoder
    """

    def __init__(self, cfg, out_stride):
        super(Decoder, self).__init__()
        self.activate_f = cfg.MODEL.ACTIVATION_F
        self.use_aspp = cfg.MODEL.USE_ASPP
        BxF = cfg.MODEL.NUM_BLOCKS * cfg.MODEL.FILTER_MULTIPLIER
        inp = 32
        rates = cfg.MODEL.ASPP_RATES
        self.pre_conv = ASPPModule(inp, 32, rates, use_gap=False, activate_f=self.activate_f)
        self.proj = conv1x1_bn(BxF, 32, 1, activate_f=self.activate_f)

        # self.transformer = Transformer_vit(Dim_in=64, Crop_size=cfg.DATASET.CROP_SIZE)
        self.transformer = Transformer_levit(Dim_in=64, Crop_size=cfg.DATASET.CROP_SIZE)
        # self.transformer = Transformer_psf(Dim_in=64)

        self.pre_cls = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
        )

        self.cls = nn.Conv2d(64, cfg.DATASET.CATEGORY_NUM, kernel_size=1, stride=1)


    def forward(self, x, targets=None, loss_dict=None, loss_weight=None):
        x0, x1 = x
        x1 = self.pre_conv(x1)
        x = torch.cat((self.proj(x0), x1), dim=1).mean(dim=2)
        x = self.transformer(x)
        x = self.pre_cls(x)
        pred = self.cls(x)
        return pred


class AutoDecoder(nn.Module):
    """ ASPP Module at each output features
    """

    def __init__(self, cfg, out_strides):
        super(AutoDecoder, self).__init__()
        self.aspps = nn.ModuleList()
        self.activate_f = cfg.MODEL.ACTIVATION_F
        BxF = 64
        affine = cfg.MODEL.AFFINE
        num_strides = len(out_strides)
        for i, out_stride in enumerate(out_strides):
            rate = out_stride
            inp = 64

            oup = BxF
            self.aspps.append(ASPPModule(inp, oup, [rate], affine=affine, use_gap=False, activate_f=self.activate_f))

        self.pre_cls=nn.Sequential(
            nn.Conv2d(BxF * num_strides, BxF * num_strides, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(BxF * num_strides),
            nn.Conv2d(BxF * num_strides, BxF * num_strides, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(BxF * num_strides),
        )
        self.cls = nn.Conv2d(BxF * num_strides, cfg.DATASET.CATEGORY_NUM, kernel_size=1, stride=1)

    def forward(self, x):
        x = [aspp(x_i) for aspp, x_i in zip(self.aspps, x)]
        x = torch.cat(x, dim=1).mean(dim=2)
        x = self.pre_cls(x)
        pred = self.cls(x)
        return pred


def build_decoder(cfg, out_strides=[2, 4, 8, 16]):
    """
    out_stride (int or List)
    """
    if cfg.SEARCH.SEARCH_ON:
        out_strides = np.ones(cfg.MODEL.NUM_STRIDES, np.int16) * 2
        return AutoDecoder(cfg, out_strides)
    else:
        return Decoder(cfg, out_strides)
