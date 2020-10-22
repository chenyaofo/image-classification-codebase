import time
import torch
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
import torch.optim as optim

import logging
import enum
import functools
import utils
from utils.distributed import world_size
from utils.metrics import AccuracyMetric, AverageMetric, EstimatedTimeArrival


def fetch(datas):
    # print(type(datas))
    if isinstance(datas[0], dict):
        images, targets = datas[0]["data"], datas[0]["label"]
        targets = targets.squeeze().long()
    else:
        images, targets = datas
    images, targets = images.cuda(non_blocking=True), targets.cuda(non_blocking=True)
    return images, targets


class SpeedTester():
    def __init__(self):
        self.reset()

    def reset(self):
        self.batch_size = 0
        self.start = time.perf_counter()

    def update(self, tensor):
        batch_size, *_ = tensor.shape
        self.batch_size += batch_size
        self.end = time.perf_counter()

    def compute(self):
        if self.batch_size == 0:
            return 0
        else:
            return self.batch_size/(self.end-self.start)


def train(epoch, model, loader, critirion, optimizer, scheduler, report_freq):
    model.train()
    if hasattr(loader, "_size"):
        loader_len = loader._size//loader.batch_size
        loader.reset()
    else:
        loader_len = len(loader)
        loader.sampler.set_epoch(epoch)
    loss_metric = AverageMetric()
    accuracy_metric = AccuracyMetric(topk=(1, 5))
    ETA = EstimatedTimeArrival(loader_len)
    speed_tester = SpeedTester()

    if scheduler is not None:
        scheduler.step(epoch)

    utils.logger.info(f"Train start, epoch={epoch:04d}, lr={optimizer.param_groups[0]['lr']:.6f}")

    for iter_, datas in enumerate(loader):
        inputs, targets = fetch(datas)

        outputs = model(inputs)
        loss = critirion(outputs, targets)

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        loss_metric.update(loss)
        accuracy_metric.update(outputs, targets)
        ETA.step()
        speed_tester.update(inputs)

        if iter_ % report_freq == 0 or iter_ == loader_len-1:
            utils.logger.info(", ".join([
                "Train",
                f"epoch={epoch:04d}",
                f"iter={iter_:05d}/{loader_len:05d}",
                f"speed={speed_tester.compute()*world_size():.2f} images/s",
                f"loss={loss_metric.compute():.4f}",
                f"top1-accuracy={accuracy_metric.at(1).rate*100:.2f}%",
                f"top5-accuracy={accuracy_metric.at(5).rate*100:.2f}%",
                f"ETA={ETA.remaining_time}",
                f"cost={ETA.cost_time}" if iter_ == loader_len - 1 else "",
            ]))
            speed_tester.reset()

    return loss_metric.compute(), (accuracy_metric.at(1).rate, accuracy_metric.at(5).rate)


def evaluate(epoch, model, loader, critirion, optimizer, scheduler, report_freq):
    model.eval()

    if hasattr(loader, "_size"):
        loader_len = loader._size//loader.batch_size
        loader.reset()
    else:
        loader_len = len(loader)
        loader.sampler.set_epoch(epoch)
    loss_metric = AverageMetric()
    accuracy_metric = AccuracyMetric(topk=(1, 5))
    ETA = EstimatedTimeArrival(loader_len)
    speed_tester = SpeedTester()

    for iter_, datas in enumerate(loader):
        inputs, targets = fetch(datas)
        with torch.no_grad():
            outputs = model(inputs)
            loss = critirion(outputs, targets)

        loss_metric.update(loss)
        accuracy_metric.update(outputs, targets)
        ETA.step()
        speed_tester.update(inputs)

        if iter_ % report_freq == 0 or iter_ == loader_len-1:
            utils.logger.info(", ".join([
                "EVAL",
                f"epoch={epoch:04d}",
                f"iter={iter_:05d}/{loader_len:05d}",
                f"speed={speed_tester.compute()*world_size():.2f} images/s",
                f"loss={loss_metric.compute():.4f}",
                f"top1-accuracy={accuracy_metric.at(1).rate*100:.2f}%",
                f"top5-accuracy={accuracy_metric.at(5).rate*100:.2f}%",
                f"ETA={ETA.remaining_time}",
                f"cost={ETA.cost_time}" if iter_ == loader_len - 1 else "",
            ]))
            speed_tester.reset()

    return loss_metric.compute(), (accuracy_metric.at(1).rate, accuracy_metric.at(5).rate)
