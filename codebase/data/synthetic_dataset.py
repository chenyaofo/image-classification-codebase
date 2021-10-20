from typing_extensions import runtime
import logging

import torch
from .register import DATA

_logger = logging.getLogger(__name__)


class SyntheticDataLoader:
    def __init__(self, input_size, target_size, num_classes,  device="cuda"):
        self.images = torch.rand(input_size, device=device, dtype=torch.float)
        self.targets = torch.randint(0, num_classes, target_size, device=device, dtype=torch.long)

    def __iter__(self):
        self.n = 0
        return self

    def __next__(self):
        self.n += 1
        if self.n == len(self):
            raise StopIteration
        return self.images, self.targets

    def __len__(self):
        return 100


@DATA.register
def synthetic_data(input_size, target_size, num_classes, device, **kwargs):
    return SyntheticDataLoader(input_size, target_size, num_classes, device),\
        SyntheticDataLoader(input_size, target_size, num_classes, device)
