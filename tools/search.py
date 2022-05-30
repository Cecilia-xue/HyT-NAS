"""
Searching script
"""
import argparse
import torch
import os
import sys
sys.path.append('..')
from one_stage_nas.config import cfg
from one_stage_nas.data import build_dataset
from one_stage_nas.solver import make_lr_scheduler
from one_stage_nas.solver import make_optimizer
from one_stage_nas.engine.searcher import do_search
from one_stage_nas.modeling.architectures import build_model
from one_stage_nas.utils.checkpoint import Checkpointer
from one_stage_nas.utils.logger import setup_logger
from one_stage_nas.utils.misc import mkdir
from tensorboardX import SummaryWriter

def search(cfg, output_dir):
    model = build_model(cfg)
    optimizer = make_optimizer(cfg, model)
    scheduler = make_lr_scheduler(cfg, optimizer)

    checkpointer = Checkpointer(
        model, optimizer, scheduler, output_dir + '/models', save_to_disk=True)

    train_loaders, val_loaders = build_dataset(cfg)

    arguments = {}
    arguments["epoch"] = 0

    extra_checkpoint_data = checkpointer.load(cfg.MODEL.WEIGHT)
    arguments.update(extra_checkpoint_data)

    # just use data parallel
    model = torch.nn.DataParallel(model).cuda()

    checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD
    val_period = cfg.SOLVER.VALIDATE_PERIOD
    max_epoch = cfg.SOLVER.MAX_EPOCH
    arch_start_epoch = cfg.SEARCH.ARCH_START_EPOCH

    writer = SummaryWriter(logdir=output_dir + '/log', comment=cfg.DATASET.DATA_SET)

    do_search(
        model,
        train_loaders,
        val_loaders,
        max_epoch,
        arch_start_epoch,
        val_period,
        optimizer,
        scheduler,
        checkpointer,
        checkpoint_period,
        arguments,
        writer,
        cfg,
        visual_dir=output_dir,
    )


def main():
    parser = argparse.ArgumentParser(description="neural architecture search for four different low-level tasks")
    parser.add_argument(
        "--config-file",
        default='../configs/HoustonU/search_ad.yaml',
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    parser.add_argument(
        "--device",
        default='5',
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
    #os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    output_dir = '/'.join((cfg.OUTPUT_DIR, '{}/Outline-{}c{}n_TC-{}'.
                           format(cfg.DATASET.DATA_SET, cfg.MODEL.NUM_LAYERS, cfg.MODEL.NUM_BLOCKS,
                                  cfg.SEARCH.TIE_CELL), 'search'))
    mkdir(output_dir)
    mkdir(output_dir + '/models')
    logger = setup_logger("one_stage_nas", output_dir)
    logger.info(args)

    logger.info("Loaded configuration file {}".format(args.config_file))
    with open(args.config_file, "r") as cf:
        config_str = "\n" + cf.read()
        logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))

    search(cfg, output_dir)


if __name__ == "__main__":
    main()
