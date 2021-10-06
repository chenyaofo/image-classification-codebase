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

from codebase.torchutils.common import set_proper_device
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


def main_worker(local_rank: int,
                ngpus_per_node: int,
                args: Args,
                conf: ConfigTree):
    device = set_proper_device(local_rank)

    rank = args.node_rank*ngpus_per_node+local_rank
    init_logger(rank=rank, filenmae=args.output_dir/"default.log")
    writer = SummaryWriter(args.output_dir) if is_master() else DummyClass()

    # log some diagnostic messages
    if not conf.get_bool("only_evaluate"):
        _logger.info("Collect envs from system:\n" + get_pretty_env_info())
        _logger.info("Args:\n" + pprint.pformat(dataclasses.asdict(args)))

    # init distribited
    if args.world_size > 1:
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=rank)

    # init model, optimizer, scheduler
    model_config = conf.get("model")
    load_from = model_config.pop("load_from")
    model: nn.Module = MODEL.build_from(model_config)
    if load_from is not None:
        model.load_state_dict(torch.load(conf.get("model.load_from"), map_location="cpu"))

    if is_dist_avail_and_init() and conf.get_bool("sync_batchnorm"):
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)

    image_size = conf.get_int('data.image_size')
    _logger.info(f"Model details: n_params={compute_nparam(model)/1e6:.2f}M, "
                 f"flops={compute_flops(model,(1,3, image_size, image_size))/1e6:.2f}M.")

    train_loader, val_loader = DATA.build_from(conf.get("data"))

    criterion = CRITERION.build_from(conf.get("criterion"))
    optimizer = OPTIMIZER.build_from(conf.get("optimizer"), dict(params=model.parameters()))
    scheduler = SCHEDULER.build_from(conf.get("scheduler"), dict(optimizer=optimizer))

    if torch.cuda.is_available():
        model = model.to(device=device)
        criterion = criterion.to(device=device)

    # restore metrics, model, optimizer and scheduler state of the checkpoint
    metrics = MetricsList()
    minitor_metric = "val/top1_acc"
    states = dict(model=unwarp_module(model), optimizer=optimizer, scheduler=scheduler)
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

    if conf.get_bool("only_evaluate"):
        val_metrics = evaluate(
            epoch=0,
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            log_interval=conf.get_int("log_interval")
        )
        _logger.info(f"EVAL complete, top1-acc={val_metrics['val/top1_acc']*100:.2f}%, " +
                     f"top5-acc={val_metrics['val/top5_acc']*100:.2f}%")
    else:
        ETA = EstimatedTimeArrival(conf.get_int("max_epochs"))
        for epoch in range(start_epoch+1, conf.get_int("max_epochs")+1):
            if is_dist_avail_and_init():
                if hasattr(train_loader, "sampler"):
                    train_loader.sampler.set_epoch(epoch)
                    val_loader.sampler.set_epoch(epoch)

            metrics += train(
                epoch=epoch,
                model=model,
                loader=train_loader,
                criterion=criterion,
                optimizer=optimizer,
                scheduler=scheduler,
                use_amp=conf.get_bool("use_amp"),
                accmulated_steps=conf.get_int("accmulated_steps"),
                device=device,
                log_interval=conf.get_int("log_interval")
            )
            metrics += evaluate(
                epoch=epoch,
                model=model,
                loader=val_loader,
                criterion=criterion,
                device=device,
                log_interval=conf.get_int("log_interval")
            )

            # record metric in tensorboard
            for name, metric_values in metrics.items():
                writer.add_scalar(name, metric_values[-1], epoch)

            # save checkpoint
            saver.save(minitor=minitor_metric, metrics=metrics.as_plain_dict(), states=states)

            ETA.step()

            # log the best metric
            best_epoch_index, _ = find_best_metric(metrics[minitor_metric])
            best_top1acc = metrics["val/top1_acc"][best_epoch_index]
            best_top5acc = metrics["val/top5_acc"][best_epoch_index]
            _logger.info(f"Epoch={epoch:04d} complete, best val top1-acc={best_top1acc*100:.2f}%, "
                         f"top5-acc={best_top5acc*100:.2f}% (epoch={best_epoch_index+1}), {ETA}")


if __name__ == "__main__":
    patch_download_in_cn()

    main(get_args())
