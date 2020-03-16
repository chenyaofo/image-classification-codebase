import torch.optim as optim
from .warmup_cosine_annealing import WarmupCosineAnnealingLR


def cosine_annealing(optimizer, args):
    return optim.lr_scheduler.CosineAnnealingLR(optimizer, args.max_epochs, args.eta_min)


def warmup_cosine(optimizer, args):
    return WarmupCosineAnnealingLR(optimizer, args.warmup_epochs,  args.max_epochs, args.eta_min)


def build_scheduler(optimizer, args):
    maps = dict(
        # cosine_annealing=cosine_annealing,
        warmup_cosine=warmup_cosine
    )
    return maps[args.scheduler](optimizer, args)
