from typing_extensions import runtime
import logging

import torch
from .register import DATA

_logger = logging.getLogger(__name__)


class SyntheticDataLoader:
    def __init__(self, image_size, target_size, device="cuda"):
        self.images = torch.rand(image_size, device=device, dtype=torch.float)
        self.targets = torch.rand(target_size, device=device, dtype=torch.long)

    def __iter__(self):
        return self

    def __next__(self):
        return self.images, self.targets

    def __len__(self):
        return 10 ^ 6


@DATA.register
def synthetic_data(image_size, target_size, device, **kwargs):
    return SyntheticDataLoader(image_size, target_size, device),\
        SyntheticDataLoader(image_size, target_size, device)
