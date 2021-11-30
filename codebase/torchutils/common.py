import os
import logging
import time
import copy
import socket
import typing
import inspect
import random
import subprocess
import pathlib
import collections
from functools import partial, reduce, wraps

import numpy
import torch
import torch.hub as hub
import torch.nn as nn
import torch.optim as optim
import torch.cuda.amp as amp
import torch.backends.cudnn

from .distributed import torchsave, is_master

_logger = logging.getLogger(__name__)


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
    _logger.info(f"Set the training job to be reproducible with seed={seed} "
                 "(more detailed can be found at https://pytorch.org/docs/stable/notes/randomness.html)")


def set_cudnn_auto_tune() -> None:
    """A wrap to set torch.backends.cudnn.benchmark to True.
    """
    torch.backends.cudnn.benchmark = True
    _logger.info(f"Set torch.backends.cudnn.benchmark=True")


def disable_debug_api() -> None:
    torch.autograd.set_detect_anomaly(False)
    torch.autograd.profiler.profile(False)
    torch.autograd.profiler.emit_nvtx(False)
    _logger.info(f"Disable the debug api for better performance: torch.autograd.set_detect_anomaly, "
                 "torch.autograd.profiler.profile, and torch.autograd.profiler.emit_nvtx")


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
    def size_hook(module: nn.Module, input: torch.Tensor, output: torch.Tensor, name: str):
        module.input_size = input[0].shape
        module.output_size = output.shape

    def prod(items):
        return reduce(lambda a, b: a*b, items)

    hooks = []
    for name, m in module.named_modules():
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            hooks.append(m.register_forward_hook(partial(size_hook, name=name)))

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
            *_, h, w = m.output_size
            kh, kw = m.kernel_size
            flops += h * w * m.in_channels * m.out_channels * kh * kw / m.groups
        if isinstance(m, nn.Linear):
            flops += prod(m.input_size) * m.out_features
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
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        _host, port = s.getsockname()
    return port


class GradientAccumulator:
    def __init__(self, steps=1, enabled=True):
        self.steps = steps
        self.enable = enabled
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

    def backward_step(self, model: nn.Module, loss: torch.Tensor,
                      optimizer: optim.Optimizer, scaler: amp.GradScaler):
        if not self.enable:
            return
        if optimizer is None:
            return

        loss = loss / self.steps

        if self.is_start_cycle:
            # if pytorch version >= 1.7, set set_to_none=True for better performance
            optimizer.zero_grad(set_to_none=True)

        if isinstance(model, nn.parallel.DistributedDataParallel) and not self.is_end_cycle:
            with model.no_sync():
                scaler.scale(loss).backward()
        else:
            scaler.scale(loss).backward()

        if self.is_end_cycle:
            scaler.step(optimizer)
            scaler.update()

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


def patch_download_in_cn():
    download_url_to_file = hub.download_url_to_file

    def _cn_download_url_to_file(url: str, dst, hash_prefix=None, progress=True):
        if url.startswith("https://github.com"):
            cdn_url = "https://dl.chenyf.workers.dev/" + url
            return download_url_to_file(cdn_url, dst, hash_prefix, progress)
    hub.download_url_to_file = _cn_download_url_to_file


class MetricsStore(collections.defaultdict):
    def __init__(self, dominant_metric_name: str = None, max_is_best: bool = True):
        super(MetricsStore, self).__init__(list)
        self.dominant_metric_name = dominant_metric_name
        self.max_is_best = max_is_best
        self.try_to_find = max if self.max_is_best else min

    def set_dominant_metric(self, dominant_metric: str):
        self.dominant_metric_name = dominant_metric

    @property
    def best_epoch(self) -> int:
        try:
            dominant_metrics = self.get(self.dominant_metric_name)
        except KeyError:
            raise KeyError("The dominant_metric_name={dominant_metric_name} are not found in the store.")
        return dominant_metrics.index(self.try_to_find(dominant_metrics))

    def get_best_metrics(self, key_filter=lambda x: True):
        return self._get_metrics(self.best_epoch, key_filter)

    def get_last_metrics(self, key_filter=lambda x: True):
        return self._get_metrics(-1, key_filter)

    def is_best_epoch(self):
        return self.best_epoch == len(self) - 1

    def _get_metrics(self, index, key_filter=lambda x: True):
        return {k: v[index] for k, v in self.items() if key_filter(k)}

    def __add__(self, another_metrics: dict):
        for k, v in another_metrics.items():
            self[k].append(v)
        return self

    @property
    def total_epoch(self):
        for k, v in self.items():
            return len(v)
        return 0

    def as_plain_dict(self):
        return {k: v for k, v in self.items()}


class StateCheckPoint:
    def __init__(self, output_directory: pathlib.Path, checkpoint_name: str = "checkpoint.pt") -> None:
        self.output_directory = output_directory
        self.checkpoint_name = checkpoint_name
        self.best_checkpoint_name = f"best_{self.checkpoint_name}"

    def is_ckpt_exists(self):
        return (self.output_directory / self.checkpoint_name).exists()

    @only_master
    def save(self, metric_store: MetricsStore, states: dict):
        checkpoint = {k: v.state_dict() for k, v in states.items() if hasattr(v, "state_dict")}
        checkpoint["metrics"] = metric_store.as_plain_dict()

        torchsave(checkpoint, self.output_directory / self.checkpoint_name)

        if metric_store.is_best_epoch():
            os.link(self.output_directory / self.checkpoint_name, self.output_directory / self.best_checkpoint_name)

    def restore(self, metric_store: MetricsStore, states: dict, device="cuda:0"):
        checkpoint_path = self.output_directory / self.checkpoint_name
        if checkpoint_path.exists():
            map_location = f"cuda:{device}" if isinstance(device, int) else device
            checkpoint: dict = torch.load(checkpoint_path, map_location=map_location)
            metric_store.update(checkpoint.pop("metrics", dict()))
            for name, module in states.items():
                if hasattr(module, "load_state_dict"):
                    module.load_state_dict(checkpoint[name])
            _logger.info(f"Load state checkpoint from {checkpoint_path} at epoch={metric_store.total_epoch}")


class ThroughputTester():
    def __init__(self):
        self.reset()

    def reset(self):
        self.batch_size = 0
        self.start = time.perf_counter()

    def update(self, tensor):
        batch_size, *_ = tensor.shape
        self.batch_size += batch_size
        self.end = time.perf_counter()

    def compute(self):
        if self.batch_size == 0:
            return 0
        else:
            return self.batch_size/(self.end-self.start)


class time_enumerate:
    def __init__(self, seq, start=0):
        self.seq = seq
        self.start = start
        self.counter = self.start-1

    def __iter__(self):
        self.seq_iter = iter(self.seq)
        return self

    def __next__(self):
        while True:
            start_time = time.perf_counter()
            item = next(self.seq_iter)
            end_time = time.perf_counter()
            self.counter += 1
            return end_time-start_time, self.counter, item


CURRENT_DEVICE = None


def set_proper_device(local_rank: int):
    global CURRENT_DEVICE
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        CURRENT_DEVICE = torch.cuda.current_device()
    else:
        CURRENT_DEVICE = "cpu"


def get_device():
    global CURRENT_DEVICE
    return CURRENT_DEVICE
