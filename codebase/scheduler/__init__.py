import torch.optim as optim
from .warmup_cosine_annealing import WarmupCosineAnnealingLR
from .warmup_exponential import WarmupExponentialLR

def cosine_annealing(optimizer, args):
    return optim.lr_scheduler.CosineAnnealingLR(optimizer, args.max_epochs, args.eta_min)


def warmup_cosine(optimizer, args):
    return WarmupCosineAnnealingLR(optimizer, args.warmup_epochs, args.max_epochs, args.eta_min)

def warmup_exponential_wrap(optimizer, args):
    return WarmupExponentialLR(optimizer, args.warmup_epochs, args.exponential_lr_lambda)


def build_scheduler(optimizer, args):
    maps = dict(
        # cosine_annealing=cosine_annealing,
        warmup_cosine=warmup_cosine,
        warmup_exponential=warmup_exponential_wrap,
    )
    return maps[args.scheduler](optimizer, args)
