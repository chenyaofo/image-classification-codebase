import math
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


def fetch_data(datas, device):
    if isinstance(datas, (list, tuple)):
        if isinstance(datas[0], dict):
            inputs = datas[0]["images"]
            targets = datas[0]["targets"].squeeze(-1).long()
        else:
            inputs, targets = datas
    else:
        raise ValueError(f"Invilid data format with data={datas}")
    inputs, targets = inputs.to(device=device, non_blocking=True), targets.to(device=device, non_blocking=True)
    return inputs, targets


def train(epoch: int,
          model: nn.Module,
          loader: data.DataLoader,
          criterion: nn.modules.loss._Loss,
          optimizer: optim.Optimizer,
          scheduler: optim.lr_scheduler._LRScheduler,
          #   only_epoch_sche: bool,
          use_amp: bool,
          accmulated_steps: int,
          device: str,
          log_interval: int):
    model.train()

    scaler = GradScaler() if use_amp else None

    gradident_accumulator = GradientAccumulator(accmulated_steps)

    if hasattr(loader, "length"):
        loader_len = loader.length
    else:
        loader_len = len(loader)

    time_cost_metric = AverageMetric("time_cost")
    loss_metric = AverageMetric("loss")
    accuracy_metric = AccuracyMetric(topk=(1, 5))
    ETA = EstimatedTimeArrival(loader_len)
    speed_tester = SpeedTester()

    if scheduler is not None:
        scheduler.step(epoch)

    lr = optimizer.param_groups[0]['lr']
    _logger.info(f"Train start, epoch={epoch:04d}, lr={lr:.6f}")

    cnt = 0

    for time_cost, iter_, datas in time_enumerate(loader, start=1):
        inputs, targets = fetch_data(datas, device)

        optimizer.zero_grad()

        with autocast(enabled=use_amp):
            outputs = model(inputs)
            loss: torch.Tensor = criterion(outputs, targets)

        gradident_accumulator.backward_step(model, loss, optimizer, scaler)

        time_cost_metric.update(time_cost)
        loss_metric.update(loss)
        accuracy_metric.update(outputs, targets)
        ETA.step()
        speed_tester.update(inputs)

        batch_size, *_ = inputs.shape
        cnt += batch_size
        # _logger.info(f"train total batch size={cnt}")

        if iter_ % log_interval == 0 or iter_ == loader_len:
            _logger.info(", ".join([
                "TRAIN",
                f"epoch={epoch:04d}",
                f"iter={iter_:05d}/{loader_len:05d}",
                f"fetch data time cost={time_cost_metric.compute()*1000:.2f}ms",
                f"fps={speed_tester.compute()*world_size():.0f} images/s",
                f"{loss_metric}",
                f"{accuracy_metric}",
                f"{ETA}",
            ]))
            time_cost_metric.reset()
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

    if hasattr(loader, "length"):
        loader_len = loader.length
    else:
        loader_len = len(loader)

    time_cost_metric = AverageMetric("time_cost")
    loss_metric = AverageMetric("loss")
    accuracy_metric = AccuracyMetric(topk=(1, 5))
    ETA = EstimatedTimeArrival(loader_len)
    speed_tester = SpeedTester()
    cnt = 0

    for time_cost, iter_, datas in time_enumerate(loader, start=1):
        inputs, targets = fetch_data(datas, device)

        batch_size, *_ = inputs.shape
        cnt += batch_size
        # _logger.info(f"eval total batch size={cnt}")

        with torch.no_grad():
            outputs = model(inputs)
            loss = criterion(outputs, targets)

        time_cost_metric.update(time_cost)
        loss_metric.update(loss)
        accuracy_metric.update(outputs, targets)
        ETA.step()
        speed_tester.update(inputs)

        if iter_ % log_interval == 0 or iter_ == loader_len:
            _logger.info(", ".join([
                "EVAL",
                f"epoch={epoch:04d}",
                f"iter={iter_:05d}/{loader_len:05d}",
                f"fetch data time cost={time_cost_metric.compute()*1000:.2f}ms",
                f"fps={speed_tester.compute()*world_size():.0f} images/s",
                f"{loss_metric}",
                f"{accuracy_metric}",
                f"{ETA}",
            ]))
            speed_tester.reset()
            time_cost_metric.reset()

    return {
        "val/loss": loss_metric.compute(),
        "val/top1_acc": accuracy_metric.at(1).rate,
        "val/top5_acc": accuracy_metric.at(5).rate,
    }
