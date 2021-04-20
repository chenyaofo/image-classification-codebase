import logging

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from torch.cuda.amp import autocast, GradScaler

from codebase.torchutils.distributed import world_size
from codebase.torchutils.metrics import AccuracyMetric, AverageMetric, EstimatedTimeArrival
from codebase.torchutils.common import GradientAccumulator
from codebase.torchutils.common import SpeedTester, time_enumerate

_logger = logging.getLogger(__name__)


def train(epoch: int,
          model: nn.Module,
          loader: data.DataLoader,
          criterion: nn.modules.loss._Loss,
          optimizer: optim.Optimizer,
          scheduler: optim.lr_scheduler._LRScheduler,
          only_epoch_sche: bool,
          use_amp: bool,
          accmulated_steps: int,
          device: str,
          log_interval: int):
    model.train()

    scaler = GradScaler() if use_amp else None

    gradident_accumulator = GradientAccumulator(accmulated_steps)

    loss_metric = AverageMetric("loss")
    accuracy_metric = AccuracyMetric(topk=(1, 5))
    ETA = EstimatedTimeArrival(len(loader))
    speed_tester = SpeedTester()

    lr = optimizer.param_groups[0]['lr']
    _logger.info(f"Train start, epoch={epoch:04d}, lr={lr:.6f}")

    for time_cost, iter_, (inputs, targets) in time_enumerate(loader, start=1):
        inputs, targets = inputs.to(device=device), targets.to(device=device)

        optimizer.zero_grad()

        with autocast(enabled=use_amp):
            outputs = model(inputs)
            loss: torch.Tensor = criterion(outputs, targets)

        gradident_accumulator.backward_step(model, loss, optimizer, scaler)

        if scheduler is not None:
            if only_epoch_sche:
                if iter_ == 1:
                    scheduler.step()
            else:
                scheduler.step()

        loss_metric.update(loss)
        accuracy_metric.update(outputs, targets)
        ETA.step()
        speed_tester.update(inputs)

        if iter_ % log_interval == 0 or iter_ == len(loader):
            _logger.info(", ".join([
                "TRAIN",
                f"epoch={epoch:04d}",
                f"iter={iter_:05d}/{len(loader):05d}",
                f"fetch data time cost={time_cost*1000:.2f}ms",
                f"fps={speed_tester.compute()*world_size():.0f} images/s",
                f"{loss_metric}",
                f"{accuracy_metric}",
                f"{ETA}",
            ]))
            speed_tester.reset()

    return {
        "lr": lr,
        "train/loss": loss_metric.compute(),
        "train/top1_acc": accuracy_metric.at(1).rate,
        "train/top5_acc": accuracy_metric.at(5).rate,
    }


def evaluate(epoch: int,
             model: nn.Module,
             loader: data.DataLoader,
             criterion: nn.modules.loss._Loss,
             device: str,
             log_interval: int):
    model.eval()

    loss_metric = AverageMetric("loss")
    accuracy_metric = AccuracyMetric(topk=(1, 5))
    ETA = EstimatedTimeArrival(len(loader))
    speed_tester = SpeedTester()

    for time_cost, iter_, (inputs, targets) in time_enumerate(loader, start=1):
        inputs, targets = inputs.to(device=device), targets.to(device=device)

        with torch.no_grad():
            outputs = model(inputs)
            loss = criterion(outputs, targets)

        loss_metric.update(loss)
        accuracy_metric.update(outputs, targets)
        ETA.step()
        speed_tester.update(inputs)

        if iter_ % log_interval == 0 or iter_ == len(loader):
            _logger.info(", ".join([
                "EVAL",
                f"epoch={epoch:04d}",
                f"iter={iter_:05d}/{len(loader):05d}",
                f"fetch data time cost={time_cost*1000:.2f}ms",
                f"fps={speed_tester.compute()*world_size():.0f} images/s",
                f"{loss_metric}",
                f"{accuracy_metric}",
                f"{ETA}",
            ]))
            speed_tester.reset()

    return {
        "val/loss": loss_metric.compute(),
        "val/top1_acc": accuracy_metric.at(1).rate,
        "val/top5_acc": accuracy_metric.at(5).rate,
    }
