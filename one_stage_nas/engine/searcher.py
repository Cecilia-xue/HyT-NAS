import os
import logging
import time
import datetime

import torch
import matplotlib.pyplot as plt
from one_stage_nas.utils.metric_logger import MetricLogger
from one_stage_nas.utils.comm import reduce_loss_dict
from one_stage_nas.utils.visualize import model_visualize
from one_stage_nas.engine.inference import inference


def do_search(
        model,
        train_loaders,
        val_loaders,
        max_epoch,
        arch_start_epoch,
        val_period,
        optimizer,
        scheduler,
        checkpointer,
        checkpointer_period,
        arguments,
        writer,
        cfg,
        visual_dir):
    """
    num_classes (int): number of classes. Required by computing mIoU.
    """
    logger = logging.getLogger("one_stage_nas.searcher")
    logger.info("Start searching")

    start_epoch = arguments["epoch"]
    start_training_time = time.time()

    val_best_acc, val_min_loss = 0, 10
    for epoch in range(start_epoch, max_epoch):
        epoch = epoch + 1
        arguments["epoch"] = epoch

        scheduler.step()

        train(model, train_loaders, optimizer, epoch, train_arch=epoch > arch_start_epoch)
        if epoch > cfg.SEARCH.ARCH_START_EPOCH:
            save_dir = '/'.join((visual_dir, 'visualize', 'arch_epoch{}'.format(epoch)))
            model_visualize(model, save_dir, cfg.SEARCH.TIE_CELL)
        if epoch % val_period == 0:
            acc, loss = inference(model, val_loaders)
            if acc > val_best_acc:
                val_best_acc, val_min_loss = acc, loss
                checkpointer.save("model_best", **arguments)
            elif acc == val_best_acc and loss < val_min_loss:
                val_best_acc, val_min_loss = acc, loss
                checkpointer.save("model_best", **arguments)
            print('val_acc:{}% val_loss:{}'.format(acc, loss))
            writer.add_scalars('Search_acc', { 'val_acc': acc}, epoch)
            writer.add_scalars('Search_loss', {'val_loss': loss}, epoch)
        if epoch % checkpointer_period == 0:
            checkpointer.save("model_{:03d}".format(epoch), **arguments)
        if epoch == max_epoch:
            checkpointer.save("model_final", **arguments)

    total_training_time = time.time() - start_training_time
    total_time_str = str(datetime.timedelta(seconds=total_training_time))
    logger.info("Total training time: {}".format(total_time_str))


def train(model, data_loaders, optimizer, epoch,
          train_arch=False):
    """
    Should add some stats and log to visualise the archs
    """
    data_loader_w = data_loaders[0]
    data_loader_a = data_loaders[1]
    optim_w = optimizer['optim_w']
    optim_a = optimizer['optim_a']

    logger = logging.getLogger("one_stage_nas.searcher")

    max_iter = len(data_loader_w)
    model.train()
    meters = MetricLogger(delimiter="  ")
    end = time.time()
    for iteration, (images, targets) in enumerate(data_loader_w):
        data_time = time.time() - end

        if train_arch:
            images_a, targets_a = next(iter(data_loader_a))
            loss = model(images_a, targets_a)
            optim_a.zero_grad()
            #loss.backward()
            loss.backward(torch.ones_like(loss))
            optim_a.step()


        loss = model(images, targets)
        # print('iteration:{} loss:{}'.format(iteration, float(loss)))
        meters.update(loss=loss)
        optim_w.zero_grad()
        #loss.backward()
        loss.backward(torch.ones_like(loss))
        optim_w.step()

        batch_time = time.time() - end
        end = time.time()
        meters.update(time=batch_time, data=data_time)

        eta_seconds = meters.time.global_avg * (max_iter - iteration)
        eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))

        if iteration % 50 == 0:
            logger.info(
                meters.delimiter.join(
                ["eta: {eta}",
                 "iter: {epoch}/{iter}",
                 "{meters}",
                 "lr: {lr:.6f}"]).format(
                     eta=eta_string,
                     epoch=epoch,
                     iter=iteration,
                     meters=str(meters),
                     lr=optim_w.param_groups[0]['lr']))
