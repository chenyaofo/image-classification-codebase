import math


class WarmupExponentialLR(object):
    def __init__(self, optimizer, T_warmup, lambda_, last_epoch=-1):
        self.T_warmup = T_warmup
        self.lambda_ = lambda_

        self.optimizer = optimizer

        self.base_lr = optimizer.param_groups[0]["lr"]

    def step(self, epoch):
        if epoch < self.T_warmup:
            lr = self.base_lr * epoch / self.T_warmup
        else:
            lr = self.base_lr * self.lambda_**(epoch - self.T_warmup)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
