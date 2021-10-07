import torch.optim as optim

from .register import OPTIMIZER

# OPTIMIZER.register(optim.SGD)
# OPTIMIZER.register(optim.Adam)
# OPTIMIZER.register(optim.LBFGS)


@OPTIMIZER.register
def CustomSGD(params, lr, momentum=0, dampening=0, weight_decay=0, nesterov=False, bn_weight_decay=None, **kwargs):
    def add_bn_extra_ophp(model, extra_ophp):
        # add extra optimizer hyper-parameters for bn layers
        basic_params = [v for n, v in params if not "bn" in n]
        bn_params = [v for n, v in params if "bn" in n]

        basic_params_ophp = dict(params=basic_params)
        bn_params_ophp = {**dict(params=bn_params), **extra_ophp}
        return [basic_params_ophp, bn_params_ophp]
    if bn_weight_decay is None:
        return optim.SGD([v for n, v in params], lr, momentum, dampening, weight_decay, nesterov)
    else:
        return optim.SGD(add_bn_extra_ophp(params, dict(weight_decay=bn_weight_decay)),
                         lr, momentum, dampening, weight_decay, nesterov)
