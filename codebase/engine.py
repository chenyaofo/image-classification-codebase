import logging

import torch
from torch.cuda.amp import autocast, GradScaler

from codebase.torchutils.distributed import world_size
from codebase.torchutils.metrics import AccuracyMetric, AverageMetric, EstimatedTimeArrival
from codebase.torchutils.common import SpeedTester, time_enumerate

_logger = logging.getLogger(__name__)


def train(epoch, model, loader, criterion, optimizer, scheduler,
          use_amp, device, log_interval):
    model.train()

    if use_amp:
        scaler = GradScaler()

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
            loss = criterion(outputs, targets)

        if use_amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

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

    if scheduler is not None:
        scheduler.step()

    return {
        "lr": lr,
        "train/loss": loss_metric.compute(),
        "train/top1_acc": accuracy_metric.at(1).rate,
        "train/top5_acc": accuracy_metric.at(5).rate,
    }


def evaluate(epoch, model, loader, criterion, device, log_interval):
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
