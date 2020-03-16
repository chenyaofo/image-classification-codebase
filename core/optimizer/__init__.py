import torch.optim as optim


def build_optimizer(params, args):
    maps = dict(
        sgd=sgd
    )
    return maps[args.optimizer](params, args)


def sgd(params, args):
    return optim.SGD(params=params, lr=args.lr, momentum=args.momentum,
                     weight_decay=args.weight_decay)
