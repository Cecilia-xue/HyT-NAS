"""
Searching script
"""

import argparse
from tensorboardX import SummaryWriter

import torch
import os
import sys
sys.path.append('..')
from one_stage_nas.config import cfg
from one_stage_nas.data import build_dataset
from one_stage_nas.solver import make_lr_scheduler
from one_stage_nas.solver import make_optimizer
from one_stage_nas.engine.trainer import do_train
from one_stage_nas.modeling.architectures import build_model
from one_stage_nas.utils.checkpoint import Checkpointer
from one_stage_nas.utils.logger import setup_logger
from one_stage_nas.utils.misc import mkdir
from one_stage_nas.utils.visualize import visualize
from one_stage_nas.engine.inference import inference

def train(cfg, output_dir):
    model = build_model(cfg)

    # visualize
    visual_dir = output_dir + '/arch'
    geno_cell = model.genotype()
    visualize(geno_cell, visual_dir, cfg.SEARCH.TIE_CELL)

    # just use data parallel
    model = torch.nn.DataParallel(model).cuda()



    # model.eval()
    # input = torch.randn(2, 1, 103, 32, 32).cuda()
    # target = torch.randint(0, 9, (2, 32, 32)).cuda()
    # model(input, target)

    optimizer = make_optimizer(cfg, model)
    scheduler = make_lr_scheduler(cfg, optimizer)

    checkpointer = Checkpointer(
        model, optimizer, scheduler, output_dir + '/models', save_to_disk=True)

    train_loader, val_loader = build_dataset(cfg)

    # best_model_dir = "/home/ren/hkzhang/code/3D_ANAS_2/tools/output/Pavia/Outline-4c3n_TC-False/train/models/model_best.pth"
    # model.load_state_dict(torch.load(best_model_dir)['model'])
    # val_acc, val_loss = inference(model, val_loader)

    arguments = {}
    arguments["iteration"] = 0
    arguments["genotype"] = model.module.genotype()

    extra_checkpoint_data = checkpointer.load(cfg.MODEL.WEIGHT)
    arguments.update(extra_checkpoint_data)

    checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD
    val_period = cfg.SOLVER.VALIDATE_PERIOD
    max_iter = cfg.SOLVER.TRAIN.MAX_ITER

    writer = SummaryWriter(logdir=output_dir + '/log', comment=cfg.DATASET.DATA_SET)

    do_train(
        model,
        train_loader,
        val_loader,
        max_iter,
        val_period,
        optimizer,
        scheduler,
        checkpointer,
        checkpoint_period,
        arguments,
        writer,
        cfg
    )


def main():

    parser = argparse.ArgumentParser(description="One-stage NAS Training")
    parser.add_argument(
        "--config-file",
        default="../configs/Pavia/train_spe.yaml",
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    parser.add_argument(
        "--device",
        default='3',
        help="path to config file",
        type=str,
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.device
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()


    output_dir = '/'.join((cfg.OUTPUT_DIR, '{}/Outline-{}c{}n_TC-{}'.
                           format(cfg.DATASET.DATA_SET, cfg.MODEL.NUM_LAYERS,
                                  cfg.MODEL.NUM_BLOCKS, cfg.SEARCH.TIE_CELL), 'train'))


    mkdir(output_dir+'/models')

    logger = setup_logger("one_stage_nas", output_dir)
    logger.info(args)

    logger.info("Loaded configuration file {}".format(args.config_file))
    with open(args.config_file, "r") as cf:
        config_str = "\n" + cf.read()
        logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))

    train(cfg, output_dir)


if __name__ == "__main__":
    main()
