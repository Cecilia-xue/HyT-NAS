# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from bisect import bisect_right
import math

import torch
from torch import nn, optim


# FIXME ideally this would be achieved with a CombinedLRScheduler,
# separating MultiStepLR with WarmupLR
# but the current LRScheduler design doesn't allow it
class WarmupMultiStepLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(
        self,
        optimizer,
        milestones,
        gamma=0.1,
        warmup_factor=1.0 / 3,
        warmup_iters=500,
        warmup_method="linear",
        last_epoch=-1,
    ):
        if not list(milestones) == sorted(milestones):
            raise ValueError(
                "Milestones should be a list of" " increasing integers. Got {}",
                milestones,
            )

        if warmup_method not in ("constant", "linear"):
            raise ValueError(
                "Only 'constant' or 'linear' warmup_method accepted"
                "got {}".format(warmup_method)
            )
        self.milestones = milestones
        self.gamma = gamma
        self.warmup_factor = warmup_factor
        self.warmup_iters = warmup_iters
        self.warmup_method = warmup_method
        super(WarmupMultiStepLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        warmup_factor = 1
        if self.last_epoch < self.warmup_iters:
            if self.warmup_method == "constant":
                warmup_factor = self.warmup_factor
            elif self.warmup_method == "linear":
                alpha = self.last_epoch / self.warmup_iters
                warmup_factor = self.warmup_factor * (1 - alpha) + alpha
        return [
            base_lr
            * warmup_factor
            * self.gamma ** bisect_right(self.milestones, self.last_epoch)
            for base_lr in self.base_lrs
        ]


class PolynormialLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(
        self,
        optimizer,
        max_iter,
        power=0.9,
        last_epoch=-1,
    ):
        self.max_iter = max_iter
        self.power = power
        super(PolynormialLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        return [
            base_lr
            * (1 - self.last_epoch / self.max_iter) ** self.power
            for base_lr in self.base_lrs
        ]


class PolyCosineAnnealingLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, max_iter, T_max, eta_min=0, power=0.9, last_epoch=-1):
        self.max_iter = max_iter
        self.power = power
        self.T_max = T_max
        self.eta_min = eta_min
        super(PolyCosineAnnealingLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        return [self.eta_min + (base_lr - self.eta_min) *
                (1 + math.cos(math.pi * self.last_epoch / self.T_max)) / 2
                * (1 - self.last_epoch / self.max_iter) ** self.power
                for base_lr in self.base_lrs]


# Compare with legacy implementation
class LegacyCosineAnnealingLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, T_max, eta_min=0, last_epoch=-1):
        self.T_max = T_max
        self.eta_min = eta_min
        super(LegacyCosineAnnealingLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        return [self.eta_min + (base_lr - self.eta_min) *
                (1 + math.cos(math.pi * self.last_epoch / self.T_max)) / 2
                for base_lr in self.base_lrs]


def main():
    # test cosine
    # Test new scheduler
    print("test new scheduler")
    model = nn.Linear(10, 2)
    optimizer = optim.SGD(model.parameters(), lr=1.)
    steps = 10
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, steps)

    try:
        for epoch in range(2):
            for idx in range(steps):
                scheduler.step()
                print(scheduler.get_lr())
                print(optimizer.param_groups[0]['lr'])
    except ZeroDivisionError as e:
        print(e)

    print("test old scheduler")
    model = nn.Linear(10, 2)
    optimizer = optim.SGD(model.parameters(), lr=1.)
    steps = 10
    scheduler = LegacyCosineAnnealingLR(optimizer, steps)

    for epoch in range(2):
        for idx in range(steps):
            scheduler.step()
            print(scheduler.get_lr())
            print(optimizer.param_groups[0]['lr'])

if __name__ == "__main__":
    main()
