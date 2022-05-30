"""
DARTS operations
"""
import torch.nn as nn
import torch

# from DCNv2.dcn_v2 import DCN


OPS = {
    'none' : lambda C, affine: Zero(),
    'max_p_3' : lambda C, affine: nn.MaxPool3d(3, stride=1, padding=1),
    'skip_connect' : lambda C, affine: Identity(),

    'con_3-3' : lambda C, affine: LeakyConvBN(C, C, 3, 3, affine=affine),
    'con_5-3' : lambda C, affine: LeakyConvBN(C, C, 5, 3, affine=affine),
    'con_3-5' : lambda C, affine: LeakyConvBN(C, C, 3, 5, affine=affine),
    'con_5-5' : lambda C, affine: LeakyConvBN(C, C, 5, 5, affine=affine),
    'con_7-7' : lambda C, affine: LeakyConvBN(C, C, 7, 7, affine=affine),
    'sep_3-3' : lambda C, affine: LeakySepConv(C, C, 3, 3, affine=affine),
    'sep_5-3' : lambda C, affine: LeakySepConv(C, C, 5, 3, affine=affine),
    'sep_3-5' : lambda C, affine: LeakySepConv(C, C, 3, 5, affine=affine),
    'sep_5-5' : lambda C, affine: LeakySepConv(C, C, 5, 5, affine=affine),
    'sep_7-7' : lambda C, affine: LeakySepConv(C, C, 7, 7, affine=affine),

    'con_3' : lambda C, affine: LeakyConvBN_2(C, C, 3, 3, affine=affine),
    'con_5' : lambda C, affine: LeakyConvBN_2(C, C, 5, 5, affine=affine),
    'con_7' : lambda C, affine: LeakyConvBN_2(C, C, 7, 7, affine=affine),
    'sep_3' : lambda C, affine: LeakySepConv_2(C, C, 3, 3, affine=affine),
    'sep_5' : lambda C, affine: LeakySepConv_2(C, C, 5, 5, affine=affine),
    'sep_7' : lambda C, affine: LeakySepConv_2(C, C, 7, 7, affine=affine),

    'udcon_3-3' : lambda C, affine: LeakyConvBN_2(C, C, 3, 3, affine=affine),
    'udcon_5-3' : lambda C, affine: LeakyConvBN_2(C, C, 5, 3, affine=affine),
    'udcon_3-5' : lambda C, affine: LeakyConvBN_2(C, C, 3, 5, affine=affine),
    'udsep_3-3' : lambda C, affine: LeakySepConv_2(C, C, 3, 3, affine=affine),
    'udsep_5-3' : lambda C, affine: LeakySepConv_2(C, C, 5, 3, affine=affine),
    'udsep_3-5' : lambda C, affine: LeakySepConv_2(C, C, 3, 5, affine=affine),

    'acon_3-1' : lambda C, affine: LeakyConvBN(C, C, 3, 1, affine=affine),
    'acon_5-1' : lambda C, affine: LeakyConvBN(C, C, 5, 1, affine=affine),
    'econ_1-3' : lambda C, affine: LeakyConvBN(C, C, 1, 3, affine=affine),
    'econ_1-5' : lambda C, affine: LeakyConvBN(C, C, 1, 5, affine=affine),
    'esep_1-3' : lambda C, affine: LeakySepConv(C, C, 1, 3, affine=affine),
    'esep_1-5' : lambda C, affine: LeakySepConv(C, C, 1, 5, affine=affine),
    'asep_3-1' : lambda C, affine: LeakySepConv(C, C, 3, 1, affine=affine),
    'asep_5-1' : lambda C, affine: LeakySepConv(C, C, 5, 1, affine=affine),
}

class LeakyConvBN(nn.Module):
    """not used"""

    def __init__(self, C_in, C_out, Spa_s, Spe_s, affine=True):
        super(LeakyConvBN, self).__init__()
        self.op = nn.Sequential(
            nn.LeakyReLU(negative_slope=0.2, inplace=False),
            nn.Conv3d(C_in, C_out, (1, Spa_s, Spa_s), padding=(0, Spa_s//2, Spa_s//2), bias=False),
            nn.Conv3d(C_out, C_out, (Spe_s, 1, 1), padding=(Spe_s//2, 0, 0), bias=False),
            nn.BatchNorm3d(C_out, affine=affine),
        )

    def forward(self, x):
        return self.op(x)


class LeakySepConv(nn.Module):

    def __init__(self, C_in, C_out, Spa_s, Spe_s, affine=True, repeats=2):
        super(LeakySepConv, self).__init__()
        basic_op = lambda: nn.Sequential(
          nn.LeakyReLU(negative_slope=0.2, inplace=False),
          nn.Conv3d(C_in, C_in, (1, Spa_s, Spa_s), padding=(0, Spa_s // 2, Spa_s // 2), groups=C_in, bias=False),
          nn.Conv3d(C_in, C_in, (Spe_s, 1, 1), padding=(Spe_s // 2, 0, 0), groups=C_in, bias=False),
          nn.Conv3d(C_in, C_out, kernel_size=1, padding=0, bias=False),
          nn.BatchNorm3d(C_out, affine=affine),
        )
        self.op = nn.Sequential()
        for idx in range(repeats):
            self.op.add_module('sep_{}'.format(idx),
                               basic_op())

    def forward(self, x):
        return self.op(x)



class LeakyConvBN_2(nn.Module):
    """not used"""

    def __init__(self, C_in, C_out, Spa_s, Spe_s, affine=True):
        super(LeakyConvBN_2, self).__init__()
        self.op = nn.Sequential(
            nn.LeakyReLU(negative_slope=0.2, inplace=False),
            nn.Conv3d(C_in, C_out, (Spe_s, Spa_s, Spa_s), padding=(Spe_s//2, Spa_s//2, Spa_s//2), bias=False),
            nn.BatchNorm3d(C_out, affine=affine),
        )

    def forward(self, x):
        return self.op(x)


class LeakySepConv_2(nn.Module):

    def __init__(self, C_in, C_out, Spa_s, Spe_s, affine=True, repeats=2):
        super(LeakySepConv_2, self).__init__()
        basic_op = lambda: nn.Sequential(
          nn.LeakyReLU(negative_slope=0.2, inplace=False),
          nn.Conv3d(C_in, C_in, (Spe_s, Spa_s, Spa_s), padding=(Spe_s // 2, Spa_s // 2, Spa_s // 2), groups=C_in, bias=False),
          nn.Conv3d(C_in, C_out, kernel_size=1, padding=0, bias=False),
          nn.BatchNorm3d(C_out, affine=affine),
        )
        self.op = nn.Sequential()
        for idx in range(repeats):
            self.op.add_module('sep_{}'.format(idx),
                               basic_op())

    def forward(self, x):
        return self.op(x)


class Identity(nn.Module):

    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class Zero(nn.Module):

    def __init__(self):
        super(Zero, self).__init__()

    def forward(self, x):
        return x.mul(0.)


