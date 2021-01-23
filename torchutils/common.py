import os
import copy
import socket
import typing
import inspect
import random
import subprocess
from functools import wraps

import numpy
import torch
import torch.nn as nn
import torch.optim as optim

from .distributed import torchsave, is_master


def is_running_in_openpai() -> bool:
    """

    Returns:
        bool: True if run in OpenPAI platform, False otherwise.
    """
    # refer to 'https://openpai.readthedocs.io/en/latest/manual/cluster-user/how-to-use-advanced-job-settings.html#environmental-variables-and-port-reservation'
    return "PAI_USER_NAME" in os.environ


def generate_random_seed() -> int:
    """Generate ranom integer number from /dev/urandom, ranged from [0, 2^16].

    Returns:
        int: A random integer number.
    """
    return int.from_bytes(os.urandom(2), byteorder="little", signed=False)


def set_reproducible(seed: int = 0) -> None:
    """To ensure the reproducibility, refer to https://pytorch.org/docs/stable/notes/randomness.html.
    Note that completely reproducible results are not guaranteed.

    Args:
        seed (int, optional): The seed. Defaults to 0.
    """
    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def set_cudnn_auto_tune() -> None:
    """A wrap to set torch.backends.cudnn.benchmark to True.
    """
    torch.backends.cudnn.benchmark = True


def compute_nparam(module: nn.Module) -> int:
    """Count how many parameter in a module. Note that the buffer in the module will not
    be counted.

    Args:
        module (nn.Module): The module to be counted.

    Returns:
        int: The number of parameters in the module.
    """
    return sum(map(lambda p: p.numel(), module.parameters()))


def compute_flops(module: nn.Module, size: int) -> int:
    """Compute the #MAdds of a module. The current version of this function can only compute the
    #MAdds of nn.Conv2d and nn.Linear. Besides, the input of the module should be a single tensor.

    Args:
        module (nn.Module): The module to be computed.
        size (int): The size of the input tensor.

    Returns:
        int: The number of MAdds.
    """
    def size_hook(module: nn.Module, input: torch.Tensor, output: torch.Tensor):
        *_, h, w = output.shape
        module.output_size = (h, w)
    hooks = []
    for name, m in module.named_modules():
        if isinstance(m, nn.Conv2d):
            hooks.append(m.register_forward_hook(size_hook))
    with torch.no_grad():
        training = module.training
        module.eval()
        module(torch.rand(size))
        module.train(mode=training)
    for hook in hooks:
        hook.remove()

    flops = 0
    for name, m in module.named_modules():
        if isinstance(m, nn.Conv2d):
            h, w = m.output_size
            kh, kw = m.kernel_size
            flops += h * w * m.in_channels * m.out_channels * kh * kw / m.groups
        if isinstance(module, nn.Linear):
            flops += m.in_features * m.out_features
    return flops


def get_last_commit_id() -> str:
    """Get the last commit id by calling the git command in another process.

    Returns:
        str: The last commit id if this folder is initialized by git, None otherwise.
    """
    try:
        commit_id = subprocess.check_output(
            ['git', 'rev-parse', 'HEAD']).decode("utf-8")
        commit_id = commit_id.strip()
        return commit_id
    except subprocess.CalledProcessError as e:
        out_bytes = e.output
        code = e.returncode
        return None


def get_branch_name() -> str:
    """Get the current brach name by calling the git command in another process.

    Returns:
        str: The current brach name if this folder is initialized by git, None otherwise.
    """
    try:
        branch_name = subprocess.check_output(
            ['git', 'symbolic-ref', '--short', '-q', 'HEAD']).decode("utf-8")
        branch_name = branch_name.strip()
        return branch_name
    except subprocess.CalledProcessError as e:
        out_bytes = e.output
        code = e.returncode
        return None


def get_gpus_memory_info() -> typing.List[typing.Tuple[int, int]]:
    """Query the GPU memory info in this server. It will call 'nvidia-smi' in another process.

    Returns:
        typing.List[typing.Tuple[int, int]]: A list of different GPU memory info. Each item in the list
        is a tuple. The tuple contrains two numbers, the first is the used memory (MB) in this GPU card, 
        the second is the toal memory (MB) in this GPU card.
    """
    try:
        query_rev = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=memory.used,memory.total', '--format=csv,noheader']).decode("utf-8")
        gpus_memory_info = []
        for item in query_rev.split("\n")[:-1]:
            gpus_memory_info.append(tuple(map(lambda x: int(x.replace(" MiB", "")), item.split(","))))
        return gpus_memory_info
    except subprocess.CalledProcessError as e:
        out_bytes = e.output
        code = e.returncode
        return None


def get_free_port() -> int:
    """

    Returns:
        int: A free port in the machine.
    """
    sock = socket.socket()
    sock.bind(('', 0))
    ip, port = sock.getnameinfo()
    sock.close()
    return port


def save_checkpoint(output_directory: str, epoch: int, model: nn.Module, optimizer: optim.Optimizer,
                    best_acc1: float, best_acc5: float, best_epoch: int) -> None:
    """Save a checkpoint and the best model in the history.

    Args:
        output_directory (str): The output directory.
        epoch (int): The epoch of the current checkpoint.
        model (nn.Module): The model in current epoch.
        optimizer (optim.Optimizer): The optimizer in current epoch.
        best_acc1 (float): The best top-1 accuracy of the model in the history.
        best_acc5 (float): The best top-5 accuracy of the model in the history.
        best_epoch (int): The eopch of the best top-1 accuracy in the history.
    """
    if isinstance(model, (nn.DataParallel, nn.parallel.DistributedDataParallel)):
        model_without_parallel = model.module
    else:
        model_without_parallel = model
    ckpt = dict(
        epoch=epoch,
        state_dict=model_without_parallel.state_dict(),
        optimizer=optimizer.state_dict(),
        best_acc1=best_acc1,
        best_acc5=best_acc5,
    )
    torchsave(ckpt, os.path.join(output_directory, "checkpoint.pth"))
    if epoch == best_epoch:
        torchsave(ckpt, os.path.join(output_directory, "best.pth"))


class GradientAccumulator:
    def __init__(self, steps=1):
        self.steps = steps
        self._counter = 0

    @property
    def counter(self):
        return self._counter

    def inc_counter(self):
        self._counter += 1
        self._counter %= self.steps

    @property
    def is_start_cycle(self):
        return self._counter == 0

    @property
    def is_end_cycle(self):
        return self._counter == self.steps - 1

    def bw_step(self, loss: torch.Tensor, optimizer: optim.Optimizer):
        if optimizer is None:
            return

        loss.backward(gradient=1/self.steps)
        if self.is_start_cycle:
            optimizer.zero_grad()
        if self.is_end_cycle:
            optimizer.step()

        self.inc_counter()


def dummy_func(*args, **kargs):
    pass


class DummyClass:
    def __getattribute__(self, obj):
        return dummy_func


class FakeObj:
    def __getattr__(self, name):
        return do_nothing


def do_nothing(*args, **kwargs) -> FakeObj:
    return FakeObj()


def only_master_fn(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        if is_master() or kwargs.get('run_anyway', False):
            kwargs.pop('run_anyway', None)
            return fn(*args, **kwargs)
        else:
            return FakeObj()

    return wrapper


def only_master_cls(cls):
    for key, value in cls.__dict__.items():
        if callable(value):
            setattr(cls, key, only_master_fn(value))

    return cls


def only_master_obj(obj):
    cls = obj.__class__
    for key, value in cls.__dict__.items():
        if callable(value):
            obj.__dict__[key] = only_master_fn(value).__get__(obj, cls)

    return obj


def only_master(something):
    if inspect.isfunction(something):
        return only_master_fn(something)
    elif inspect.isclass(something):
        return only_master_cls(something)
    else:
        return only_master_obj(something)


class ModelEma(nn.Module):
    """ 
    Copy from https://github.com/rwightman/pytorch-image-models/blob/master/timm/utils/model_ema.py#L82
    License under Apache License 2.0, full license text can be found at 
    https://github.com/rwightman/pytorch-image-models/blob/master/LICENSE
    """

    def __init__(self, model, decay=0.9999, device=None):
        super(ModelEma, self).__init__()
        # make a copy of the model for accumulating moving average of weights
        self.module = copy.deepcopy(model)
        self.module.eval()
        self.decay = decay
        self.device = device  # perform ema on different device from model if set
        if self.device is not None:
            self.module.to(device=device)

    def _update(self, model, update_fn):
        with torch.no_grad():
            for ema_v, model_v in zip(self.module.state_dict().values(), model.state_dict().values()):
                if self.device is not None:
                    model_v = model_v.to(device=self.device)
                ema_v.copy_(update_fn(ema_v, model_v))

    def update(self, model):
        self._update(model, update_fn=lambda e, m: self.decay * e + (1. - self.decay) * m)

    def set(self, model):
        self._update(model, update_fn=lambda e, m: m)


def unwarp_module(model):
    if isinstance(model, (nn.parallel.DataParallel, nn.parallel.DistributedDataParallel, ModelEma)):
        return model.module
    else:
        return model
