import torch
from torch import nn
import torch.nn.functional as F

from one_stage_nas.darts.cell import CellSPE, CellSPA
from one_stage_nas.darts.genotypes import PRIMITIVES
from .decoders import build_decoder
from .common import conv_bn
from .loss import CELoss


class Identity(nn.Module):

    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x



class HSI_supernet(nn.Module):
    def __init__(self, cfg):
        super(HSI_supernet, self).__init__()
        self.f = cfg.MODEL.FILTER_MULTIPLIER
        self.num_layers = cfg.MODEL.NUM_LAYERS
        self.num_blocks = cfg.MODEL.NUM_BLOCKS
        self.primitives_spa = PRIMITIVES[cfg.MODEL.PRIMITIVES_SPA]
        self.primitives_spe = PRIMITIVES[cfg.MODEL.PRIMITIVES_SPE]
        self.activatioin_f = cfg.MODEL.ACTIVATION_F
        self.use_res = cfg.MODEL.USE_RES
        affine = cfg.MODEL.AFFINE
        #self.stem1 = conv_bn(1, 32, (5, 3, 3), (2, 1, 1), (0, 1, 1), affine)
        self.stem1 = conv_bn(1, 32, (5, 3, 3), (1, 1, 1), (2, 1, 1), affine)
        self.stem2 = conv_bn(32,32, (3, 3, 3), (1, 1, 1), (1, 1, 1), affine)

        self.cells_spe = nn.ModuleList()
        self.cells_spa = nn.ModuleList()
        self.cell_routers_spe = nn.ModuleList()
        self.cell_routers_spa = nn.ModuleList()
        self.cell_configs = []

        for l in range(1, self.num_layers + 1):
            self.cells_spe.append(CellSPE(self.num_blocks, 8,self.primitives_spe,affine=affine))
            self.cells_spa.append(CellSPA(self.num_blocks, 8,self.primitives_spa,affine=affine))
            if l==1:
                self.cell_routers_spe.append(conv_bn(32, 8, (1, 1, 1), (1, 1, 1), (0, 0, 0), affine=affine, activate_f=None))
                self.cell_routers_spa.append(conv_bn(32, 8, (1, 1, 1), (1, 1, 1), (0, 0, 0), affine=affine, activate_f=None))
            else:
                self.cell_routers_spe.append(conv_bn(64, 8, (1, 1, 1), (1, 1, 1), (0, 0, 0), affine=affine, activate_f=None))
                self.cell_routers_spa.append(conv_bn(64, 8, (1, 1, 1), (1, 1, 1), (0, 0, 0), affine=affine, activate_f=None))


        # ASPP
        self.decoder = build_decoder(cfg)
        self.init_alphas()
        self.criteria = CELoss(ignore_lb=-1)


    def w_parameters(self):
        return [value for key, value in self.named_parameters()
                if 'arch' not in key and value.requires_grad]

    def a_parameters(self):
        a_params = [value for key, value in self.named_parameters() if 'arch' in key]
        return a_params

    def init_alphas(self):
        k = sum(2 + i for i in range(self.num_blocks))
        num_ops = len(self.primitives_spa)
        self.arch_alphas = nn.Parameter(torch.ones(self.num_layers, k, num_ops))
        self.arch_betas = nn.Parameter(torch.ones(self.num_layers, k, num_ops))
        self.arch_gammas = nn.Parameter(torch.ones(self.num_layers,2))
        self.score_func = F.softmax

    def scores(self):
        return (self.score_func(self.arch_alphas, dim=-1),
                self.score_func(self.arch_betas, dim=-1),
                self.score_func(self.arch_gammas, dim=-1),)

    def forward(self, images, targets=None):

        alphas, betas,gammas = self.scores()

        # The first layer is different
        inputs_1 = self.stem1(images)
        inputs_0 = self.stem2(F.leaky_relu(inputs_1, negative_slope=0.2) )
        hidden_states = []
        for l in range(self.num_layers):
            cell_weights_spe = alphas[l]
            cell_weights_spa = betas[l]
            cell_weights_arch= gammas[l]
            if l==0:
                inputs_1 = self.cell_routers_spe[l](inputs_1)
            inputs_0 = self.cell_routers_spa[l](inputs_0)
            spe_out=self.cells_spe[l](inputs_0, inputs_1, cell_weights_spe)*cell_weights_arch[0]
            spa_out=self.cells_spa[l](inputs_0, inputs_1, cell_weights_spa)*cell_weights_arch[1]
            fused_out= torch.cat((spe_out,spa_out), dim=1)
            hidden_states.append(fused_out)
            inputs_1=inputs_0
            inputs_0=fused_out
        pred = self.decoder(hidden_states)

        if self.training:
            loss = self.criteria(pred, targets)
            return loss
        else:
            return pred

    def genotype(self):
        alphas, betas,gamma = self.scores()

        gene_cell = []
        for i in range(self.num_layers):
            if gamma[i][0]>gamma[i][1]:
                gene_cell.append(self.cells_spe[i].genotype(alphas[i]))
            else:
                gene_cell.append(self.cells_spa[i].genotype(betas[i]))
        return gene_cell
