import torch.optim as optim

from .register import SCHEDULER

from .warmup_cosine_annealing import WarmupCosineAnnealingLR

SCHEDULER.register(optim.lr_scheduler.MultiStepLR)
SCHEDULER.register(optim.lr_scheduler.CosineAnnealingLR)
SCHEDULER.register(optim.lr_scheduler.CosineAnnealingWarmRestarts)
SCHEDULER.register(optim.lr_scheduler.ExponentialLR)
SCHEDULER.register(optim.lr_scheduler.CyclicLR)
SCHEDULER.register(optim.lr_scheduler.LambdaLR)
