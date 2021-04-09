import logging
import dataclasses
import pprint

import torch
import torch.cuda
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.collect_env import get_pretty_env_info
from torch.utils.tensorboard import SummaryWriter
from pyhocon import ConfigTree

from codebase.config import Args, get_args
from codebase.data import DATA
from codebase.models import MODEL
from codebase.optimizer import OPTIMIZER
from codebase.scheduler import SCHEDULER
from codebase.criterion import CRITERION
from codebase.engine import train, evaluate

from codebase.torchutils.common import unwarp_module
from codebase.torchutils.common import compute_nparam, compute_flops
from codebase.torchutils.common import ModelSaver
from codebase.torchutils.common import MetricsList
from codebase.torchutils.common import patch_download_in_cn
from codebase.torchutils.common import DummyClass
from codebase.torchutils.common import find_best_metric
from codebase.torchutils.distributed import is_dist_avail_and_init, is_master
from codebase.torchutils.metrics import EstimatedTimeArrival
from codebase.torchutils.logging_ import init_logger


_logger = logging.getLogger(__name__)


def main(args: Args):
    distributed = args.world_size > 1
    ngpus_per_node = torch.cuda.device_count()
    if distributed:
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args, args.conf))
    else:
        local_rank = 0
        main_worker(local_rank, ngpus_per_node, args, args.conf)


def main_worker(local_rank, ngpus_per_node, args: Args, conf: ConfigTree):
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.cuda.current_device()
    else:
        device = "cpu"

    rank = args.node_rank*ngpus_per_node+local_rank
    init_logger(rank=rank, filenmae=args.output_dir/"default.log")
    writer = SummaryWriter(args.output_dir) if is_master() else DummyClass()

    _logger.info("Collect envs from system:\n" + get_pretty_env_info())
    _logger.info("Args:\n" + pprint.pformat(dataclasses.asdict(args)))

    if args.world_size > 1:
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=rank)

    model: nn.Module = MODEL.build_from(conf.get("model"))

    image_size = conf.get_int('data.image_size')
    _logger.info(f"Model details: n_params={compute_nparam(model)/1e6:.2f}M, "
                 f"flops={compute_flops(model,(1,3, image_size, image_size))/1e6:.2f}M.")

    train_loader, val_loader = DATA.build_from(conf.get("data"))

    criterion = CRITERION.build_from(conf.get("criterion"))
    optimizer = OPTIMIZER.build_from(conf.get("optimizer"), dict(params=model.parameters()))
    scheduler = SCHEDULER.build_from(conf.get("scheduler"), dict(optimizer=optimizer))

    max_epochs = conf.get_int("max_epochs")

    metrics = MetricsList()

    minitor_metric = "val/top1_acc"
    states = dict(model=unwarp_module(model), optimizer=optimizer, scheduler=scheduler)

    if torch.cuda.is_available():
        model = model.to(device=device)
        criterion = criterion.to(device=device)

    saver = ModelSaver(args.output_dir)
    if conf.get_bool("auto_resume"):
        saver.restore(metrics, states, device=device)
        start_epoch = len(metrics[minitor_metric])
        if start_epoch != 0:
            _logger.info(f"Load chckpoint from epoch={start_epoch}.")
    else:
        start_epoch = 0

    if is_dist_avail_and_init():
        model = nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])
    use_amp = conf.get_bool("use_amp")
    log_interval = conf.get_int("log_interval")

    if conf.get_bool("only_evaluate"):
        metrics += evaluate(0, model, val_loader, criterion, device, log_interval)
    else:
        ETA = EstimatedTimeArrival(max_epochs)
        for epoch in range(start_epoch+1, max_epochs+1):
            metrics += train(epoch, model, train_loader, criterion,
                             optimizer, scheduler, use_amp, device, log_interval)
            metrics += evaluate(epoch, model, val_loader, criterion, device, log_interval)

            for name, metric_values in metrics.items():
                writer.add_scalar(name, metric_values[-1], epoch)

            saver.save(minitor=minitor_metric, metrics=metrics.as_plain_dict(), states=states)
            ETA.step()
            best_epoch, best_acc = find_best_metric(metrics[minitor_metric])
            _logger.info(f"Epoch={epoch:04d} complete, best val acc={best_acc*100:.2f}% @ {best_epoch} epoch, {ETA}")


if __name__ == "__main__":
    patch_download_in_cn()

    main(get_args())
