import logging
import time
import datetime

import torch
import matplotlib.pyplot as plt

from one_stage_nas.utils.metric_logger import MetricLogger
from one_stage_nas.utils.comm import compute_params
from .inference import inference
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
        # print('pixels:{} acc:{} loss:{}'.format(self.total_pixel, self.correct_pixel, self.loss_sum))

    def reset(self):
        self.loss_sum = 0
        self.correct_pixel = 0
        self.total_pixel = 0

    def get_result(self):
        return self.correct_pixel/self.total_pixel * 100, self.loss_sum/self.total_pixel


def do_train(
        model,
        train_loader,
        val_loader,
        max_iter,
        val_period,
        optimizer,
        scheduler,
        checkpointer,
        checkpointer_period,
        arguments,
        writer,
        cfg):
    """
    num_classes (int): number of classes. Required by computing mIoU.
    """
    logger = logging.getLogger("one_stage_nas.trainer")
    logger.info("Model Params: {:.2f}K".format(compute_params(model) / 1000))

    logger.info("Start training")

    start_iter = arguments["iteration"]
    start_training_time = time.time()

    train_anal = result_analysis()

    val_best_acc, val_min_loss = 0, 10
    model.train()
    data_iter = iter(train_loader)

    meters = MetricLogger(delimiter="  ")

    end = time.time()
    for iteration in range(start_iter, max_iter):
        iteration = iteration + 1
        arguments["iteration"] = iteration

        scheduler.step()

        try:
            images, targets = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            images, targets = next(data_iter)
        data_time = time.time() - end

        pred, loss = model(images, targets)
        # meters.update(loss=loss)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(model.parameters(), 5.0)
        optimizer.step()
        train_anal(pred, targets.cuda())

        batch_time = time.time() - end
        end = time.time()
        meters.update(time=batch_time, data=data_time)

        eta_seconds = meters.time.global_avg * (max_iter - iteration)
        eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))

        if iteration % (val_period // 4) == 0:
            logger.info(
                meters.delimiter.join(
                ["eta: {eta}",
                 "iter: {iter}",
                 "{meters}",
                 "lr: {lr:.6f}",
                 "max_mem: {memory:.0f}"]).format(
                     eta=eta_string,
                     iter=iteration,
                     meters=str(meters),
                     lr=optimizer.param_groups[0]['lr'],
                     memory=torch.cuda.max_memory_allocated() / 1024.0 / 1024.0))

        if iteration % val_period == 0:
        # if iteration % 50 == 0:
            checkpointer.save("model_{:06d}".format(iteration), **arguments)
            train_acc, train_loss = train_anal.get_result()
            train_anal.reset()
            # if iteration > int(max_iter*3/4):
            # if iteration % val_period == 0:
            if True:
                val_acc, val_loss = inference(model, val_loader)
                print("val_acc:{} val_loss:{}".format(val_acc, val_loss))
                if val_acc > val_best_acc:
                    val_best_acc = val_acc
                    val_min_loss = val_loss
                    checkpointer.save("model_best", **arguments)
                elif val_acc==val_best_acc and val_loss < val_min_loss:
                    val_best_acc = val_acc
                    val_min_loss = val_loss
                    checkpointer.save("model_best", **arguments)
                model.train()
                writer.add_scalars('overall_acc', {'train_acc': train_acc, 'val_acc': val_best_acc}, iteration)
                writer.add_scalars('loss', {'train_loss': train_loss, 'val_psnr': val_min_loss}, iteration)
            else:
                writer.add_scalars('overall_acc', {'train_acc': train_acc}, iteration)
                writer.add_scalars('loss', {'train_loss': train_loss}, iteration)

        if iteration % val_period == 0:
            checkpointer.save("model_{:06d}".format(iteration), **arguments)
        if iteration == max_iter:
            checkpointer.save("model_final", **arguments)

    total_training_time = time.time() - start_training_time
    total_time_str = str(datetime.timedelta(seconds=total_training_time))
    logger.info("Total training time: {}".format(total_time_str))

    writer.close()

