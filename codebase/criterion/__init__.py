import torch.nn as nn

from .label_smooth import LabelSmoothCrossEntropyLoss


def cross_entropy_loss(args):
    return nn.CrossEntropyLoss()


def label_smooth_cross_entropy_loss(args):
    num_classes = 1000 if args.dataset == "imagenet" else 10
    return LabelSmoothCrossEntropyLoss(num_classes)


def build_criterion(args):
    maps = dict(
        ce=cross_entropy_loss,
        labelsmooth=label_smooth_cross_entropy_loss
    )
    return maps[args.criterion](args)
