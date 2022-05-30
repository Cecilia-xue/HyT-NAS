import torch
from torch import nn

from one_stage_nas.utils.comm import drop_path
from .operations import OPS, Identity


class MixedOp(nn.Module):

    def __init__(self, C, primitives, affine=True):
        super(MixedOp, self).__init__()
        self._ops = nn.ModuleList()
        for primitive in primitives:
            op = OPS[primitive](C, affine)
            self._ops.append(op)

    def forward(self, x, weights):
        return sum(w * op(x) for w, op in zip(weights, self._ops))


class CellSPE(nn.Module):
    """
    Generate outputs for next layer inside
    """

    def __init__(self, blocks, C, primitives, empty_H1=False, affine=True):
        super(CellSPE, self).__init__()

        self._steps = blocks
        self._multiplier = blocks
        self._ops = nn.ModuleList()
        self._empty_h1 = empty_H1
        self._primitives = primitives

        for i in range(self._steps):
            for j in range(2 + i):
                op = MixedOp(C, primitives, affine)
                self._ops.append(op)

        self.conv_end = nn.Sequential(
            nn.Conv3d(24, 32, (1, 1, 1), (1, 1, 1), bias=False),
            nn.BatchNorm3d(32),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )

    def forward(self, s0, s1, weights):
        states = [s1, s0]

        offset = 0
        for i in range(self._steps):
            # summing up all branches, if H1 is empty, skip the first weight
            s = sum(self._ops[offset+j](h, weights[offset+j])
                    for j, h in enumerate(states)
                    if not self._empty_h1 or j > 0)
            offset += len(states)
            # summing counting weights
            states.append(s)

        out = torch.cat(states[-self._multiplier:], dim=1)
        out = self.conv_end(out)
        return out

    def genotype(self, weights):
        """
        get cell genotype
        """
        gene = []
        n = 2
        start = 0
        for i in range(self._steps):
            end = start + n
            W = weights[start:end].clone().detach()
            edges = sorted(range(i+2),
                           key=lambda x: -max(W[x][k]
                                              for k in range(
                                                      len(W[x]))
                                              if k != self._primitives.index('none')))[:2]
            for j in edges:
                k_best = None
                for k in range(len(W[j])):
                    if k != self._primitives.index('none'):
                        if k_best is None or W[j][k] > W[j][k_best]:
                            k_best = k
                gene.append((self._primitives[k_best], j))
            start = end
            n += 1
        return gene

class CellSPA(nn.Module):
    """
    Generate outputs for next layer inside
    """

    def __init__(self, blocks, C, primitives, empty_H1=False, affine=True):
        super(CellSPA, self).__init__()

        self._steps = blocks
        self._multiplier = blocks
        self._ops = nn.ModuleList()
        self._empty_h1 = empty_H1
        self._primitives = primitives

        for i in range(self._steps):
            for j in range(2 + i):
                op = MixedOp(C, primitives, affine)
                self._ops.append(op)

        self.conv_end = nn.Sequential(
            nn.Conv3d(24, 32, (1, 1, 1), (1, 1, 1), bias=False),
            nn.BatchNorm3d(32),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )

    def forward(self, s0, s1, weights):
        states = [s1, s0]

        offset = 0
        for i in range(self._steps):
            # summing up all branches, if H1 is empty, skip the first weight
            s = sum(self._ops[offset+j](h, weights[offset+j])
                    for j, h in enumerate(states)
                    if not self._empty_h1 or j > 0)
            offset += len(states)
            # summing counting weights
            states.append(s)

        out = torch.cat(states[-self._multiplier:], dim=1)
        out = self.conv_end(out)
        return out

    def genotype(self, weights):
        """
        get cell genotype
        """
        gene = []
        n = 2
        start = 0
        for i in range(self._steps):
            end = start + n
            W = weights[start:end].clone().detach()
            edges = sorted(range(i+2),
                           key=lambda x: -max(W[x][k]
                                              for k in range(
                                                      len(W[x]))
                                              if k != self._primitives.index('none')))[:2]
            for j in edges:
                k_best = None
                for k in range(len(W[j])):
                    if k != self._primitives.index('none'):
                        if k_best is None or W[j][k] > W[j][k_best]:
                            k_best = k
                gene.append((self._primitives[k_best], j))
            start = end
            n += 1
        return gene

class FixCell(nn.Module):
    def __init__(self, genotype, C, blocks, repeats=2):
        """
        Arguments:
            genotype: cell structure
            C (int): channels, reshaped and weighted by previous cells
            empty_H1: if True, hidden state l-2 is empty

        Returns:
            h_next (Tensor BxinpxWxH): next hidden state
        """
        super(FixCell, self).__init__()
        [op_names] = genotype
        self._steps = len(op_names) // 2
        self._multiplier = self._steps
        self._ops = nn.ModuleList()
        self._indices=[]
        # self.empty_h1 = empty_H1
        for name in op_names:
            op = OPS[name[0]](C, True)
            self._ops.append(op)
            self._indices.append(name[1])

        self._indices=tuple(self._indices)
        self.conv_end = nn.Sequential(
            nn.Conv3d(24, 32, (1, 1, 1), (1, 1, 1), bias=False),
            nn.BatchNorm3d(32),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )

    def forward(self, s0, s1, drop_prob):
        states = [s1, s0]

        for i in range(self._steps):
            # summing up all branches, if H1 is empty, skip the first weight
            s = 0
            for ind in [2*i, 2*i+1]:
                # if self.empty_h1 and ind == 0: continue
                op = self._ops[ind]
                h = op(states[self._indices[ind]])
                if self.training and drop_prob > 0:
                    if not isinstance(op, Identity):
                        h = drop_path(h, drop_prob)
                s = s + h
            states.append(s)

        out = torch.cat(states[-self._multiplier:], dim=1)
        out = self.conv_end(out)
        return out
