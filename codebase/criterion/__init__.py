import torch.nn as nn

from .register import CRITERION
from .label_smooth import LabelSmoothCrossEntropyLoss

CRITERION.register(nn.CrossEntropyLoss)
