import torch.nn as nn

from .register import CRITERION


CRITERION.register(nn.CrossEntropyLoss)

load_modules(__file__)
    