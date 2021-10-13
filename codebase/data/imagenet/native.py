import os
import math
import logging
import pathlib
import warnings

from codebase.torchutils.distributed import world_size

try:
    import webdataset as wds
except ImportError:
    warnings.warn("Webdataset library is unavailable, cannot load dataset with webdataset.")

import torch.utils.data as data
import torchvision.transforms as T
from torchvision.datasets import ImageFolder

from ..utils import get_samplers, glob_tars


_logger = logging.getLogger(__name__)


def identity(x):
    return x


def get_train_transforms(crop_size, mean, std, is_training):
    pipelines = []
    if is_training:
        pipelines.append(T.RandomResizedCrop(crop_size))
        pipelines.append(T.RandomHorizontalFlip())
    else:
        pipelines.append(T.Resize(int(crop_size/7*8)))
        pipelines.append(T.CenterCrop(crop_size))
    pipelines.append(T.ToTensor())
    pipelines.append(T.Normalize(mean=mean, std=std))
    return T.Compose(pipelines)


def _build_imagenet_loader(root, is_training, image_size, mean, std, batch_size, num_workers, use_webdataset, dataset_len=None):
    transforms = get_train_transforms(image_size, mean, std, is_training)
    if use_webdataset:
        dataset = (
            wds.WebDataset(glob_tars(pathlib.Path(root)/("train" if is_training else "val")))
            .shuffle(int(os.environ.get("WDS_BUFFER_SIZE", 5000)) if is_training else -1)
            .decode("pil")
            .to_tuple("jpg;png", "cls")
            .map_tuple(transforms, identity)
            .with_length(dataset_len)
        )
        loader = wds.WebLoader(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            persistent_workers=True,
            drop_last=is_training
        )
        if is_training:
            loader = loader.ddp_equalize(len(dataset)//batch_size)
            loader = loader.with_length(loader.length)
        else:
            loader = loader.with_length(math.ceil(len(dataset)/world_size()/batch_size))
    else:
        dataset = ImageFolder(pathlib.Path(root)/("train" if is_training else "val"), transform=transforms)
        sampler = get_samplers(dataset, is_training)
        loader = data.DataLoader(dataset, batch_size=batch_size,
                                 shuffle=(sampler is None),
                                 sampler=sampler,
                                 num_workers=num_workers,
                                 persistent_workers=True,
                                 drop_last=is_training)
    _logger.info(f"Loading ImageNet dataset using torchvision from {'webdataset' if use_webdataset else 'folder'}"
                 f" with {'trainset' if is_training else 'valset'} (len={len(dataset)})")
    if use_webdataset:
        _logger.info("Note that the length of webdataset is reported by user defined config file.")
    return loader


def build_imagenet_loader(root, image_size, mean, std, batch_size, num_workers,
                          use_webdataset, trainset_len, valset_len):
    return _build_imagenet_loader(root, True, image_size, mean, std, batch_size, num_workers, use_webdataset, trainset_len),\
        _build_imagenet_loader(root, False, image_size, mean, std, batch_size, num_workers, use_webdataset, valset_len)
