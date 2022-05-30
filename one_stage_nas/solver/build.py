# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch

from .lr_scheduler import WarmupMultiStepLR, PolynormialLR
from .lr_scheduler import PolyCosineAnnealingLR


class OptimizerDict(dict):

    def __init__(self, *args, **kwargs):
        super(OptimizerDict, self).__init__(*args, **kwargs)

    def state_dict(self):
        return [optim.state_dict() for optim in self.values()]

    def load_state_dict(self, state_dicts):
        for state_dict, optim in zip(state_dicts, self.values()):
            optim.load_state_dict(state_dict)
            for state in optim.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.cuda()


def make_optimizer(cfg, model):
    if cfg.SEARCH.SEARCH_ON:
        return make_search_optimizers(cfg, model)
    else:
        return make_normal_optimizer(cfg, model)


def make_normal_optimizer(cfg, model):
    params = []
    for key, value in model.named_parameters():
        if not value.requires_grad:
            continue
        lr = cfg.SOLVER.TRAIN.INIT_LR
        weight_decay = cfg.SOLVER.WEIGHT_DECAY
        if "bias" in key:
            lr = cfg.SOLVER.TRAIN.INIT_LR * cfg.SOLVER.BIAS_LR_FACTOR
            weight_decay = cfg.SOLVER.WEIGHT_DECAY_BIAS
        params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]

    optimizer = torch.optim.SGD(params, lr, momentum=cfg.SOLVER.MOMENTUM)
    return optimizer


def make_search_optimizers(cfg, model):

    optim_w = torch.optim.SGD(model.w_parameters(),
                              lr=cfg.SOLVER.SEARCH.LR_START,
                              momentum=cfg.SOLVER.SEARCH.MOMENTUM,
                              weight_decay=cfg.SOLVER.SEARCH.WEIGHT_DECAY)
    optim_a = torch.optim.Adam(model.a_parameters(),
                               lr=cfg.SOLVER.SEARCH.LR_A,
                               weight_decay=cfg.SOLVER.SEARCH.WD_A)
    return OptimizerDict(optim_w=optim_w, optim_a=optim_a)


def make_search_lr_scheduler(cfg, optimizer_dict):
    optimizer = optimizer_dict['optim_w']

    return PolyCosineAnnealingLR(
        optimizer,
        max_iter=cfg.SOLVER.MAX_EPOCH,
        T_max=cfg.SOLVER.SEARCH.T_MAX,
        eta_min=cfg.SOLVER.SEARCH.LR_END
    )


def make_lr_scheduler(cfg, optimizer):
    if cfg.SEARCH.SEARCH_ON:
        return make_search_lr_scheduler(cfg, optimizer)
    if cfg.SOLVER.SCHEDULER == 'poly':
        power = cfg.SOLVER.TRAIN.POWER
        max_iter = cfg.SOLVER.TRAIN.MAX_ITER
        return PolynormialLR(optimizer, max_iter, power)
    return WarmupMultiStepLR(
        optimizer,
        cfg.SOLVER.STEPS,
        cfg.SOLVER.GAMMA,
        warmup_factor=cfg.SOLVER.WARMUP_FACTOR,
        warmup_iters=cfg.SOLVER.WARMUP_ITERS,
        warmup_method=cfg.SOLVER.WARMUP_METHOD,
    )
