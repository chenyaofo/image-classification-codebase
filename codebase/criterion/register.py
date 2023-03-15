import torch.nn as nn
from codebase.torchutils.register import Register

CRITERION = Register("criterion")

CRITERION.register(nn.CrossEntropyLoss)