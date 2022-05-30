"""
Discrete structure of Auto-DeepLab

Includes utils to convert continous Auto-DeepLab to discrete ones
"""

import os
import torch
from torch import nn
from torch.nn import functional as F

from one_stage_nas.darts.cell import FixCell
from .HSI_supernet import HSI_supernet
from .common import conv_bn, conv1x1_bn
from .decoders import build_decoder
from .loss import CELoss

class Identity(nn.Module):

    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


def get_genotype_from_adl(cfg):
    # create ADL model
    adl_cfg = cfg.clone()
    adl_cfg.defrost()

    adl_cfg.merge_from_list(['MODEL.META_ARCHITECTURE', 'AutoDeepLab',
                             'MODEL.FILTER_MULTIPLIER', 8,
                             'MODEL.AFFINE', True,
                             'SEARCH.SEARCH_ON', True])

    model = HSI_supernet(adl_cfg)
    # load weights
    SEARCH_RESULT_DIR = '/'.join((cfg.OUTPUT_DIR, '{}/Outline-{}c{}n_TC-{}'
                                  .format( cfg.DATASET.DATA_SET, cfg.MODEL.NUM_LAYERS, cfg.MODEL.NUM_BLOCKS,
                                           cfg.SEARCH.TIE_CELL), 'search/models/model_best.pth'))
    ckpt = torch.load(SEARCH_RESULT_DIR)
    restore = {k: v for k, v in ckpt['model'].items() if 'arch' in k}
    model.load_state_dict(restore, strict=False)
    return model.genotype()


class HSI_compnet(nn.Module):
    def __init__(self, cfg):
        super(HSI_compnet, self).__init__()

        # load genotype
        geno_file = '/'.join((cfg.OUTPUT_DIR, '{}/Outline-{}c{}n_TC-{}'.
                              format(cfg.DATASET.DATA_SET, cfg.MODEL.NUM_LAYERS, cfg.MODEL.NUM_BLOCKS,
                                     cfg.SEARCH.TIE_CELL), 'search/models/model_best.geno'))

        if os.path.exists(geno_file):
            print("Loading genotype from {}".format(geno_file))
            genotype = torch.load(geno_file, map_location=torch.device("cpu"))
        else:
            genotype = get_genotype_from_adl(cfg)
            print("Saving genotype to {}".format(geno_file))
            torch.save(genotype, geno_file)

        geno_cell = genotype

        self.genotpe = genotype

        # basic configs
        self.activate_f = cfg.MODEL.ACTIVATION_F
        self.use_res = cfg.MODEL.USE_RES
        self.f = cfg.MODEL.FILTER_MULTIPLIER
        self.num_layers = cfg.MODEL.NUM_LAYERS
        self.num_blocks = cfg.MODEL.NUM_BLOCKS
        self.in_channel = cfg.MODEL.IN_CHANNEL

        self.stem1 = conv_bn(1, 32, (5, 3, 3), (1, 1, 1), (0, 1, 1))
        self.stem2 = conv_bn(32,32, (3, 3, 3), (1, 1, 1), (1, 1, 1))
        skip_conv = []
        skip_conv.append(conv_bn(32, self.f * self.num_blocks, (3, 1, 1), (2, 1, 1), (0, 0, 0), affine=False))
        for i in range(1, self.num_layers):
            skip_conv.append(conv_bn(self.f * self.num_blocks, self.f * self.num_blocks, (3, 1, 1), (2, 1, 1), (0, 0, 0), affine=False))
        self.skip_conv = nn.Sequential(*skip_conv)
        # skip_conv.append(conv_bn(32, self.f * self.num_blocks, (3,3, 3), (1, 1, 1), (1, 1, 1), affine=False))
        # for i in range(1, self.num_layers):
        #     skip_conv.append(conv_bn(self.f * self.num_blocks, self.f * self.num_blocks, (1, 1, 1), (1, 1, 1), (0, 0, 0), affine=False))
        # self.skip_conv = nn.Sequential(*skip_conv)

        # create cells
        self.cells = nn.ModuleList()
        self.cell_routers = nn.ModuleList()
        self.scalers = nn.ModuleList()

        for layer, (geno) in enumerate(zip(geno_cell), 1):
            self.cells.append(FixCell(geno, 8, self.num_blocks))
            self.cell_routers.append(conv_bn(32, 8, (3, 1, 1), (2, 1, 1), (0, 0, 0), activate_f=None))

        self.decoder = build_decoder(cfg, out_strides=[8,8,8,8])

        self.criteria = CELoss(ignore_lb=-1)

    def genotype(self):
        return self.genotpe

    def forward(self, images, targets=None, drop_prob=-1):
        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passed.")

        h1 = self.stem1(images)
        h0 = self.stem2(F.leaky_relu(h1, negative_slope=0.2))

        endpoint = self.skip_conv(h0)

        for i, [cell, cell_router] in enumerate(zip(self.cells, self.cell_routers)):
            h1 = h0
            h0 = cell(cell_router(h0), cell_router(h1), drop_prob)

        pred = self.decoder([endpoint, F.leaky_relu(h0, negative_slope=0.2)])

        if self.training:
            loss = self.criteria(pred, targets)
            return pred, loss
        else:
            return pred


