
import torch
from torch.utils.data import DataLoader
from torchvision import transforms


def batch_normalize(tensor, mean, std, inplace=False):
    if not torch.is_tensor(tensor):
        if not tensor.ndim == 4:
            raise TypeError('tensor should be a 4D tensor.')

    if not inplace:
        tensor = tensor.clone()

    dtype = tensor.dtype

    if mean is not None:
        mean = torch.as_tensor(mean, dtype=dtype, device=tensor.device)
        tensor.sub_(mean[:, None, None])
    if std is not None:
        std = torch.as_tensor(std, dtype=dtype, device=tensor.device)
        tensor.div_(std[:, None, None])

    return tensor


class AsyncStreamDataLoader():
    def __init__(self, loader, async_stream=True, mean=None, std=None):
        self.loader = iter(loader)
        self.async_stream = async_stream
        self.mean = mean
        self.std = std
        self.stream = torch.cuda.Stream() if self.async_stream else torch.cuda.default_stream()

        self.preload()

    def preload(self):
        try:
            self.next_input, self.next_target = next(self.loader)
        except StopIteration:
            self.next_input = None
            self.next_target = None
            return

        with torch.cuda.stream(self.stream):
            self.next_input = self.next_input.cuda(non_blocking=True)
            self.next_target = self.next_target.cuda(non_blocking=True)

        self.next_input = batch_normalize(self.next_input, mean=self.mean, std=self.std)

    def next(self):
        if self.async_stream:
            torch.cuda.current_stream().wait_stream(self.stream)
            input = self.next_input
            target = self.next_target
            if input is not None:
                input.record_stream(torch.cuda.current_stream())
            if target is not None:
                target.record_stream(torch.cuda.current_stream())
        else:
            input = self.next_input
            target = self.next_target
        self.preload()
        return input, target

    def __iter__(self):
        self.loader_iter = iter(self.loader)

    def __len__(self):
        return len(self.loader)


def _to_cuda(tensor: torch.Tensor, stream: torch.cuda.Stream):
    with torch.cuda.stream(stream):
        tensor = tensor.cuda(non_blocking=True)
    torch.cuda.current_stream().wait_stream(stream)
    tensor.record_stream(torch.cuda.current_stream())
    return tensor


class AsyncStreamDataLoaderIter:
    def __init__(self, proxy: AsyncStreamDataLoader):
        self.proxy = proxy

    def __next__(self):
        data = next(self.proxy.loader_iter)
        if data
        if torch.is_tensor(data):
            data
