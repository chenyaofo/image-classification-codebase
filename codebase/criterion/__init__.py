import torch.nn as nn
from torch.nn.modules.module import Module

from .label_smooth import LabelSmoothCrossEntropyLoss
from ..config import Args
from ..register import Register

CRITERION = Register("criterion")


@CRITERION.register
def cross_entropy(args: Args) -> nn.Module:
    return nn.CrossEntropyLoss()


@CRITERION.register
def label_smooth_cross_entropy(args: Args) -> nn.Module:
    return LabelSmoothCrossEntropyLoss(args.num_classes)
