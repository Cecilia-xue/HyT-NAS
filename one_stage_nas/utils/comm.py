"""multigpu utils
"""

import torch


def reduce_loss_dict(loss_dict):
    """
    Reduce the loss dictionary from all processes so that process with rank
    0 has the averaged results. Returns a dict with the same fields as
    loss_dict, after reduction.
    """
    with torch.no_grad():
        loss_names = []
        all_losses = []
        for k in sorted(loss_dict.keys()):
            loss_names.append(k)
            all_losses.append(loss_dict[k].mean())
        reduced_losses = {k: v for k, v in zip(loss_names, all_losses)}
    return reduced_losses


def drop_path(x, drop_prob):
    if drop_prob > 0:
        keep_prob = 1 - drop_prob
        mask = torch.cuda.FloatTensor(x.size(0), 1, 1, 1).bernoulli_(keep_prob)
        x.div_(keep_prob)
        x.mul_(mask)
    return x


def compute_params(model):
    n_params = 0
    for m in model.module.parameters():
        n_params += m.numel()
    return n_params
