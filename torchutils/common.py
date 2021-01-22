import os
import socket
import inspect
import random
import subprocess
from functools import wraps

import numpy
import torch
import torch.nn as nn
import torch.optim as optim

from .distributed import torchsave, is_master


def is_running_in_openpai():
    # refer to 'https://openpai.readthedocs.io/en/latest/manual/cluster-user/how-to-use-advanced-job-settings.html#environmental-variables-and-port-reservation'
    return "PAI_USER_NAME" in os.environ

def generate_random_seed():
    return int.from_bytes(os.urandom(2), byteorder="little", signed=False)


def set_reproducible(seed=0):
    '''
    To ensure the reproducibility, refer to https://pytorch.org/docs/stable/notes/randomness.html.
    Note that completely reproducible results are not guaranteed.
    '''
    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def set_cudnn_auto_tune():
    torch.backends.cudnn.benchmark = True


def compute_nparam(module: nn.Module):
    return sum(map(lambda p: p.numel(), module.parameters()))


def compute_flops(module: nn.Module, size):
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


def get_last_commit_id():
    try:
        commit_id = subprocess.check_output(
            ['git', 'rev-parse', 'HEAD']).decode("utf-8")
        commit_id = commit_id.strip()
        return commit_id
    except subprocess.CalledProcessError as e:
        out_bytes = e.output
        code = e.returncode
        return None

def get_branch_name():
    try:
        branch_name = subprocess.check_output(
            ['git', 'symbolic-ref', '--short', '-q', 'HEAD']).decode("utf-8")
        branch_name = branch_name.strip()
        return branch_name
    except subprocess.CalledProcessError as e:
        out_bytes = e.output
        code = e.returncode
        return None

def get_gpus_memory_info():
    try:
        query_rev = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=memory.used,memory.total', '--format=csv,noheader']).decode("utf-8")
        gpus_memory_info = []
        for item in query_rev.split("\n")[:-1]:
            gpus_memory_info.append(list(map(lambda x:int(x.replace(" MiB", "")), item.split(","))))
        return gpus_memory_info
    except subprocess.CalledProcessError as e:
        out_bytes = e.output
        code = e.returncode
        return None

def get_free_port():  
    sock = socket.socket()
    sock.bind(('', 0))
    ip, port = sock.getnameinfo()
    sock.close()
    return port

def save_checkpoint(output_directory, epoch, model: nn.Module,
                    optimizer: optim.Optimizer, best_acc1, best_acc5, best_epoch):
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
