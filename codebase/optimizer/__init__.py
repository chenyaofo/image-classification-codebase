import torch.optim as optim

from .register import OPTIMIZER

OPTIMIZER.register(optim.SGD)
OPTIMIZER.register(optim.Adam)
OPTIMIZER.register(optim.LBFGS)
