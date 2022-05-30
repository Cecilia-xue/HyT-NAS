import os
import logging
import numpy as np
import torch
import torch.nn.functional as F


class result_analysis(object):
    def __init__(self):
        self.loss_sum=0
        self.correct_pixel=0
        self.total_pixel=0
    def __call__(self, pred, targets):
        N, C, H, W = pred.size()
        logits = pred.permute(0, 2, 3, 1).contiguous().view(-1, C)
        labels = targets.view(-1)
        pred_label = torch.argmax(pred, dim=1)
        loss = F.cross_entropy(logits, labels, ignore_index=-1)
        correct_pixel = float(torch.eq(pred_label, targets).sum())
        label_mask = targets > -1
        labels_num = int(label_mask.sum())
        self.loss_sum += float(loss*labels_num)
        self.correct_pixel += correct_pixel
        self.total_pixel += labels_num

    def reset(self):
        self.loss_sum = 0
        self.correct_pixel = 0
        self.total_pixel = 0

    def get_result(self):
        return self.correct_pixel/self.total_pixel * 100, self.loss_sum/self.total_pixel


def inference(model, val_loaders):
    print('start_inference')
    logger = logging.getLogger("one_stage_nas.inference")
    model.eval()
    result_anal = result_analysis()
    with torch.no_grad():
        for images, targets in val_loaders:
            pred = model(images, targets)
            result_anal(pred, targets.cuda())

        acc, loss = result_anal.get_result()
        # print(result_anal.total_pixel)
        result_anal.reset()
    return acc, loss


if __name__ == '__main__':
    result_anal = result_analysis()
    pred = torch.randn(2, 13, 32, 32).cuda(device=7)
    label = torch.randint(0, 13, [13, 32, 32]).cuda(device=7)-1
    result_anal(pred, label)

