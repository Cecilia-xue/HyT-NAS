import torch
import numpy as np
import torch.nn.functional as F
from math import exp

# PSNR
class PSNR(object):
    def __init__(self):
        self.sum_psnr = 0
        self.im_count = 0

    def __call__(self, output, gt):

        output = output*255.0
        gt = gt*255.0
        diff = (output - gt)
        mse = torch.mean(diff*diff)
        psnr = float(10*torch.log10(255.0*255.0/mse))

        self.sum_psnr = self.sum_psnr + psnr
        self.im_count += 1.0

    def metric_get(self, frac=4):
        return round(self.sum_psnr/self.im_count, frac)

    def reset(self):
        self.sum_psnr = 0
        self.im_count = 0

